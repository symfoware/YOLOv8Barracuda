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

    // 確認用の表示色 GBR -RGB
    private List<Color> _colorList = new List<Color>() {
        Color.red,     // 0: 鼻(Nose)
        Color.cyan,    // 1: 左目(Left Eye)
        Color.blue,    // 2: 右目(Right Eye)
        Color.cyan,    // 3: 左耳(Left Ear)
        Color.blue,    // 4: 右耳(Right Ear)
        Color.green,   // 5: 左肩(Left Shoulder)
        Color.magenta, // 6: 右肩(Right Shoulder)
        Color.green,   // 7: 左肘(Left Elbow)
        Color.magenta, // 8: 右肘(Right Elbow)
        Color.green,   // 9: 左手首(Left Wrist)
        Color.magenta, // 10: 右手首(Right Wrist)
        Color.gray,    // 11: 左腰(Left Hip)
        Color.yellow,  // 12: 右腰(Right Hip)
        Color.gray,    // 13: 左膝(Left Knee)
        Color.yellow,  // 14: 右膝(Right Knee)
        Color.gray,    // 15: 左足首(Left Ankle)
        Color.yellow   // 16: 右足首(Right Ankle)
    };

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
        List<DetectionResult> ditects = ParseOutputs(output0, 0.75f, 0.75f);

        worker.Dispose();
        inputTensor.Dispose();
        output0.Dispose();

        // 結果の描画
        // 縮小した画像を解析しているので、結果を元のサイズに変換
        float scaleX = _image.width / (float)_resizeLength;
        float scaleY = _image.height / (float)_resizeLength;
        // 結果表示用に画像をコピー
        var image = ResizedTexture(_image, _image.width, _image.height);
        foreach (DetectionResult ditect in ditects) {
            Debug.Log($"{ditect.score:0.00}");
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
            

            // 検出したキーポイントを描画
            int point = 0;
            foreach (KeyPoint kp in ditect.keypoints) {
                // 中心から7ピクセル範囲を描画
                int centerX = (int)(kp.x * scaleX);
                int centerY = (int)(kp.y * scaleY);
                for (int x = centerX - 3; x < centerX + 3; x++) {
                    for (int y = centerY - 3; y < centerY + 3; y++) {
                        // 検出結果は左上が原点だが、Texture2Dは左下が原点なので上下を入れ替える
                        image.SetPixel(x, _image.height-y, _colorList[point]);
                    }
                }
                point++;
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
    public float score { get; }
    public List<KeyPoint> keypoints { get; }

    public DetectionResult(Tensor t, int idx) {
        // 検出結果で得られる矩形の座標情報は0:中心x, 1:中心y、2:width, 3:height
        // 座標系を左上xy右下xyとなるよう変換
        float halfWidth = t[0, 0, idx, 2] / 2;
        float halfHeight = t[0, 0, idx, 3] / 2;
        x1 = t[0, 0, idx, 0] - halfWidth;
        y1 = t[0, 0, idx, 1] - halfHeight;
        x2 = t[0, 0, idx, 0] + halfWidth;
        y2 = t[0, 0, idx, 1] + halfHeight;
        score = t[0, 0, idx, 4];

        // 各キーポイントのx,y,visibleが設定されている
        int channels = t.shape.channels;
        keypoints = new List<KeyPoint>();
        for (int point = 5; point < channels; point+=3) {
            keypoints.Add(new KeyPoint(
                t[0, 0, idx, point],
                t[0, 0, idx, point+1],
                t[0, 0, idx, point+2]
            ));
        }
    }

}

// 検出したキーポイント情報
class KeyPoint {
    public float x { get; }
    public float y { get; }
    public float visible { get; }

    public KeyPoint(float x, float y, float visible) {
        this.x = x;
        this.y = y;
        this.visible = visible;
    }

}
