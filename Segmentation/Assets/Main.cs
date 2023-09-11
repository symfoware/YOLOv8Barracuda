using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UI = UnityEngine.UI;

public class Main : MonoBehaviour {

    public NNModel _model;
    public Texture2D _image;
    public UI.RawImage _imageView;
    
    private int _resizeLength = 640; // リサイズ後の正方形の1辺の長さ

    // Start is called before the first frame update
    void Start() {
        // onnxモデルのロードとワーカーの作成
        var model = ModelLoader.Load(_model);
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        // ラベル情報取得
        Dictionary<int, string> labels = ParseNames(model);
        
        // モデルのinputサイズに変換し、Tensorを生成
        var texture = ResizedTexture(_image, _resizeLength, _resizeLength);
        Tensor inputTensor = new Tensor(texture, channels: 3);

        // 推論実行
        worker.Execute(inputTensor);

        // 結果の解析
        Tensor output0 = worker.PeekOutput("output0");
        Tensor output1 = worker.PeekOutput("output1");
        List<DetectionResult> ditects = ParseOutputs(output0, output1, 0.5f, 0.75f);

        worker.Dispose();
        inputTensor.Dispose();
        output0.Dispose();
        output1.Dispose();

        // 結果の描画
        // 縮小した画像を解析しているので、結果を元のサイズに変換
        float scaleX = _image.width / (float)_resizeLength;
        float scaleY = _image.height / (float)_resizeLength;
        // 結果表示用に画像をコピー
        var image = ResizedTexture(_image, _image.width, _image.height);
        // 同じclassは同じ色になるように
        Dictionary<int, Color> colorMap = new Dictionary<int, Color>();

        foreach (DetectionResult ditect in ditects) {
            Debug.Log($"{labels[ditect.classId]}:{ditect.score:0.00}");
            // 領域塗りつぶし用のランダムカラー
            Color color = new Color(Random.Range(0f, 1f), Random.Range(0f, 1f), Random.Range(0f, 1f));
            if (colorMap.ContainsKey(ditect.classId)) {
                color = colorMap[ditect.classId];
            } else {
                colorMap.Add(ditect.classId, color);
            }

            // 検出した領域描画
            int x1 = (int)(ditect.x1 * scaleX);
            int x2 = (int)(ditect.x2 * scaleX);
            int y1 = (int)(ditect.y1 * scaleY);
            int y2 = (int)(ditect.y2 * scaleY);
            for (int x = x1; x < x2; x++) {
                image.SetPixel(x, _image.height-y1, Color.red);
                image.SetPixel(x, _image.height-(y1-1), Color.red);
                image.SetPixel(x, _image.height-y2, Color.red);
                image.SetPixel(x, _image.height-(y2+1), Color.red);
            }
            for (int y = y1; y < y2; y++) {
                image.SetPixel(x1, _image.height-y, Color.red);
                image.SetPixel(x1-1, _image.height-y, Color.red);
                image.SetPixel(x2, _image.height-y, Color.red);
                image.SetPixel(x2+1, _image.height-y, Color.red);
            }
            
            // 検出した矩形内をループ
            for (int x = x1; x < x2; x++) {
                for (int y = y1; y < y2; y++) {
                    // 該当座標のmaskがtrueなら、塗りつぶし実行
                    if (ditect.maskMatrix[x, y]) {
                        // 検出結果は左上が原点だが、Texture2Dは左下が原点のようなので上下を入れ替える
                        image.SetPixel(x, image.height-y, color);
                    }
                }
            }
            
        }
        image.Apply();
        _imageView.texture = image;
        
    }

    private List<DetectionResult> ParseOutputs(Tensor output0, Tensor output1, float threshold, float iouThres) {

        // 検出結果の行数
        int outputWidth = output0.shape.width;
        // 検出クラス数
        int classCount = output0.shape.channels - output1.shape.channels - 4;
        
        // 検出結果として採用する候補
        List<DetectionResult> candidateDitects = new List<DetectionResult>();
        // 使用する検出結果
        List<DetectionResult> ditects = new List<DetectionResult>();

        for (int i = 0; i < outputWidth; i++) {
            // 検出結果を解析
            var result = new DetectionResult(output0, i, classCount);
            // スコアが規定値未満なら無視
            if (result.score < threshold) {
                continue;
            }
            // 候補として追加
            candidateDitects.Add(result);
        }

        // NonMaxSuppression処理
        // 重なった矩形で最大スコアのものを採用
        while (candidateDitects.Count > 0) {
            int idx = 0;
            float maxScore = 0.0f;
            for (int i = 0; i < candidateDitects.Count; i++) {
                if (candidateDitects[i].score > maxScore) {
                    idx = i;
                    maxScore = candidateDitects[i].score;
                }
            }

            // score最大の結果を取得し、リストから削除
            var cand = candidateDitects[idx];
            candidateDitects.RemoveAt(idx);

            // 採用する結果に追加
            ditects.Add(cand);

            List<int> deletes = new List<int>();
            for (int i = 0; i < candidateDitects.Count; i++) {
                // IOUチェック
                float iou = Iou(cand, candidateDitects[i]);
                if (iou >= iouThres) {
                    deletes.Add(i);
                }
            }
            for (int i = deletes.Count - 1; i >= 0; i--) {
                candidateDitects.RemoveAt(deletes[i]);
            }

        }

        // 検出範囲のマスク処理
        ProcessMask(ditects, output1);

        return ditects;

    }

    // 物体の重なり具合判定
    private float Iou(DetectionResult boxA, DetectionResult boxB) {
        if ((boxA.x1 == boxB.x1) && (boxA.x2 == boxB.x2) && (boxA.y1 == boxB.y1) && (boxA.y2 == boxB.y2)) {
            return 1.0f;

        } else if (((boxA.x1 <= boxB.x1 && boxA.x2 > boxB.x1) || (boxA.x1 >= boxB.x1 && boxB.x2 > boxA.x1))
            && ((boxA.y1 <= boxB.y1 && boxA.y2 > boxB.y1) || (boxA.y1 >= boxB.y1 && boxB.y2 > boxA.y1))) {
            float intersection = (Mathf.Min(boxA.x2, boxB.x2) - Mathf.Max(boxA.x1, boxB.x1)) 
                * (Mathf.Min(boxA.y2, boxB.y2) - Mathf.Max(boxA.y1, boxB.y1));
            float union = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1) + (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1) - intersection;
            return (intersection / union);
        }

        return 0.0f;
    }

    // 検出結果とoutput1を利用し、マスク情報の生成
    private void ProcessMask(List<DetectionResult> ditects, Tensor output1) {
        int oh = output1.shape.height;
        int ow = output1.shape.width;
        int oc = output1.shape.channels;

        // オリジナルの画像サイズ
        int width = _image.width;
        int height = _image.height;
        float scaleX = width / (float)_resizeLength;
        float scaleY = height / (float)_resizeLength;
        
        // 解析結果のmaskとoutput1の行列を計算
        foreach(var ditect in ditects) {
            float[,] calcMatrix = new float[ow, oh];
            
            for (int i = 0; i < oc; i++) {
                float mask = ditect.masks[i];
                for (int h = 0; h < oh; h++) {
                    for (int w = 0; w < ow; w++) {
                        calcMatrix[w, h] += output1[0, h, w, i] * mask;
                    }
                }
            }

            // 結果をSigmoid適用後の値に変換
            for (int w = 0; w < ow; w++) {
                for (int h = 0; h < oh; h++) {
                    calcMatrix[w, h] = Sigmoid(calcMatrix[w, h]);
                }
            }

            // マスク情報は160x160で得られる
            // マスク情報の行列を元画像サイズに変換
            float[,] resizeMatrix = BilinearInterpolation(calcMatrix, ow, oh, width, height);

            // 検出矩形内のみのマスク情報に選別し、結果を退避
            ditect.maskMatrix = CropMask(
                resizeMatrix,
                (int)(ditect.x1 * scaleX),
                (int)(ditect.y1 * scaleY),
                (int)(ditect.x2 * scaleX),
                (int)(ditect.y2 * scaleY)
            );
        }

    }

    // シグモイド関数
    // https://www.hanachiru-blog.com/entry/2020/05/04/120000
    private float Sigmoid(float x) {
        return 1.0f / (1 + Mathf.Exp(-x));
    }
    
    // 検出結果サイズを画像サイズに合わせてリサイズ
    // 双一次補間（バイリニア補間　Bilinear）
    // https://imagingsolution.net/imaging/interpolation/
    private float[,] BilinearInterpolation(float[,] calcMatrix, int width, int height, int resizeWidth, int resizeHeight) {
        // 画像のリサイズ処理と同様
        float[,] maskMatrix = new float[resizeWidth, resizeHeight];
        float ratioX = 1.0f / ((float)resizeWidth / (width-1));
		float ratioY = 1.0f / ((float)resizeHeight / (height-1));

        for (int y = 0; y < resizeHeight; y++) {
            int yFloor = (int)Mathf.Floor(y * ratioY);
            var yLerp = y * ratioY - yFloor;
			for (int x = 0; x < resizeWidth; x++) {
				int xFloor = (int)Mathf.Floor(x * ratioX);
				var xLerp = x * ratioX - xFloor;
                
                maskMatrix[x, y] = LerpUnclamped(
                    LerpUnclamped(calcMatrix[xFloor, yFloor], calcMatrix[xFloor+1, yFloor], xLerp),
                    LerpUnclamped(calcMatrix[xFloor, yFloor+1], calcMatrix[xFloor+1, yFloor+1], xLerp),
                    yLerp
                );
            }
        }

        return maskMatrix;

    }

    private float LerpUnclamped(float m1, float m2, float value) {
        return m1 + (m2 - m1) * value;
    }

    // 検出矩形内のマスク情報を評価
    private bool[,] CropMask(float[,] maskMatrix, float fx1, float fy1, float fx2, float fy2) {
        bool[,] mask = new bool[maskMatrix.GetLength(0), maskMatrix.GetLength(1)]; // 配列の初期値はfalse
        int x1 = (int)fx1;
        int x2 = (int)fx2;
        int y1 = (int)fy1;
        int y2 = (int)fy2;
        
        // 矩形内のmask上納のみ評価
        for (int w = x1; w < x2; w++) {
            for (int h = y1; h < y2; h++) {
                // 閾値0.5以上の領域をmask対象とする
                if (0.5 <= maskMatrix[w, h]) {
                    mask[w, h] = true;
                }
            }
        }
        
        return mask;
    }


    // モデルのnames情報からラベルを取得
    private Dictionary<int, string> ParseNames(Model model) {
        Dictionary<int, string> labels = new Dictionary<int, string>();
        // {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', .. }
        // という文字列が入っているので解析
        char[] removeChars = { '{', '}', ' '};
        char[] removeCharsValue = { '\'', ' '};
        string[] items = model.Metadata["names"].Trim(removeChars).Split(",");
        foreach (string item in items) {
            string[] values =item.Split(":");
            int classId = int.Parse(values[0]);
            string name = values[1].Trim(removeCharsValue);
            labels.Add(classId, name);
        }
        
        return labels;
    }


    // 画像のリサイズ処理
    private static Texture2D ResizedTexture(Texture2D texture, int width, int height) {
        // RenderTextureに書き込む
        var rt = RenderTexture.GetTemporary(width, height);
        Graphics.Blit(texture, rt);
        // RenderTexgureから書き込む
        var preRt = RenderTexture.active;
        RenderTexture.active = rt;
        var resizedTexture = new Texture2D(width, height);
        resizedTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        resizedTexture.Apply();
        RenderTexture.active = preRt;
        RenderTexture.ReleaseTemporary(rt);
        return resizedTexture;
    }

}

// 検出結果
class DetectionResult {
    public float x1 { get; }
    public float y1 { get; }
    public float x2 { get; }
    public float y2 { get; }
    public int classId { get; }
    public float score { get; }
    public bool[,] maskMatrix { get; set; }
    public List<float> masks = new List<float>();

    public DetectionResult(Tensor t, int idx, int classCount) {
        // 検出結果で得られる矩形の座標情報は0:中心x, 1:中心y、2:width, 3:height
        // 座標系を左上xy右下xyとなるよう変換
        float halfWidth = t[0, 0, idx, 2] / 2;
        float halfHeight = t[0, 0, idx, 3] / 2;
        x1 = t[0, 0, idx, 0] - halfWidth;
        y1 = t[0, 0, idx, 1] - halfHeight;
        x2 = t[0, 0, idx, 0] + halfWidth;
        y2 = t[0, 0, idx, 1] + halfHeight;
        
        // 認識結果の確信度
        // 4: class1のscore, 5: class2のscore, 6: class3のscore...が設定されている
        // 最大値を判定して設定
        score = 0f;
        for (int i = 0; i < classCount; i++) {
            float classScore = t[0, 0, idx, i + 4];
            if (classScore < score) {
                continue;
            }
            classId = i;
            score = classScore;
        }

        // 残りのmask用情報を退避
        for (int i = classCount + 4; i < t.shape.channels; i++) {
            masks.Add(t[0, 0, idx, i]);
        }
    }

}
