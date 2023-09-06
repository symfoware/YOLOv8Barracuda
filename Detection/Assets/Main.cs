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
    // ラベルの情報
    // model.Metadata["names"]に同様の値があるが、JSON文字列で登録されており
    // 標準機能でパースできないようなので別途定義
    private readonly string[] _labels = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
        "wine glass", "cup", "fork", "knife", "spoon",  "bowl", "banana", "apple", "sandwich", "orange", 
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrus"};

    // Start is called before the first frame update
    void Start() {
        // onnxモデルのロードとワーカーの作成
        var model = ModelLoader.Load(_model);
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
        
        // モデルのinputサイズに変換し、Tensorを生成
        var texture = ResizedTexture(_image, _resizeLength, _resizeLength);
        Tensor inputTensor = new Tensor(texture, channels: 3);

        // 推論実行
        worker.Execute(inputTensor);

        // 結果の解析
        Tensor output0 = worker.PeekOutput("output0");
        List<DetectionResult> ditects = ParseOutputs(output0, 0.5f, 0.75f);

        worker.Dispose();
        inputTensor.Dispose();
        output0.Dispose();

        // 結果の描画
        // 縮小した画像を解析しているので、結果を元のサイズに変換
        float scaleX = _image.width / (float)_resizeLength;
        float scaleY = _image.height / (float)_resizeLength;
        // 結果表示用に画像をクローン
        var image = ResizedTexture(_image, _image.width, _image.height);
        // 同じclassは同じ色になるように
        Dictionary<int, Color> colorMap = new Dictionary<int, Color>();
        
        foreach (DetectionResult ditect in ditects) {
            // 解析結果表示
            Debug.Log($"{_labels[ditect.classId]}: {ditect.score:0.00}");
            
            // 領域塗りつぶし用のランダムカラー
            Color color = new Color(Random.Range(0f, 1f), Random.Range(0f, 1f), Random.Range(0f, 1f));
            if (colorMap.ContainsKey(ditect.classId)) {
                color = colorMap[ditect.classId];
            } else {
                colorMap.Add(ditect.classId, color);
            }
            
            // 検出した矩形内をループ
            for (int x = (int)(ditect.x1 * scaleX); x < (int)(ditect.x2 * scaleX); x++) {
                for (int y = (int)(ditect.y1 * scaleY); y < (int)(ditect.y2 * scaleY); y++) {
                    // 検出結果は左上が原点だが、Texture2Dは左下が原点なので上下を入れ替える
                    image.SetPixel(x, _image.height-y, color);
                }
            }
        }
        image.Apply();

        _imageView.texture = image;
        
    }

    private List<DetectionResult> ParseOutputs(Tensor output0, float threshold, float iouThres) {
        // 検出結果の行数
        int outputWidth = output0.shape.width;
        
        // 検出結果として採用する候補
        List<DetectionResult> candidateDitects = new List<DetectionResult>();
        // 使用する検出結果
        List<DetectionResult> ditects = new List<DetectionResult>();

        for (int i = 0; i < outputWidth; i++) {
            // 検出結果を解析
            var result = new DetectionResult(output0, i);
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

    public DetectionResult(Tensor t, int idx) {
        // 検出結果で得られる矩形の座標情報は0:中心x, 1:中心y、2:width, 3:height
        // 座標系を左上xy右下xyとなるよう変換
        float halfWidth = t[0, 0, idx, 2] / 2;
        float halfHeight = t[0, 0, idx, 3] / 2;
        x1 = t[0, 0, idx, 0] - halfWidth;
        y1 = t[0, 0, idx, 1] - halfHeight;
        x2 = t[0, 0, idx, 0] + halfWidth;
        y2 = t[0, 0, idx, 1] + halfHeight;

        // 残りの領域に各クラスのスコアが設定されている
        // 最大値を判定して設定
        int classes = t.shape.channels - 4;
        score = 0f;
        for (int i = 0; i < classes; i++) {
            float classScore = t[0, 0, idx, i + 4];
            if (classScore < score) {
                continue;
            }
            classId = i;
            score = classScore;
        }
    }

}
