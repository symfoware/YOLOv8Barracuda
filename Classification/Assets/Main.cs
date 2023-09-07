using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UI = UnityEngine.UI;
using TMPro;

public class Main : MonoBehaviour {

    public NNModel _model;
    public Texture2D _image;
    public UI.RawImage _imageView;
    public TextMeshProUGUI _textView;
    
    private int _resizeLength = 224; // リサイズ後の正方形の1辺の長さ

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
        Dictionary<int, float> ditects = ParseOutputs(output0, 0.1f);

        worker.Dispose();
        inputTensor.Dispose();
        output0.Dispose();

        string result = "";
        foreach (KeyValuePair<int, float> kv in ditects) {
            result += $"{labels[kv.Key]}: {kv.Value:0.00}\n";
        }

        _imageView.texture = _image;        
        _textView.text = result;
        
    }

    // モデルのnames情報からラベルを取得
    private Dictionary<int, string> ParseNames(Model model) {
        Dictionary<int, string> labels = new Dictionary<int, string>();

        // {0: 'tench', 1: 'goldfish', 2: 'great_white_shark', 3: 'tiger_shark', .. }
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

    private Dictionary<int, float> ParseOutputs(Tensor output0, float threshold) {
        // 検出結果の行数
        int outputChannels = output0.shape.channels;
        
        // 検出結果
        Dictionary<int, float> ditects = new Dictionary<int, float>();

        for (int i = 0; i < outputChannels; i++) {
            // thresholdで指定された値より低いscoreは除外
            float score = output0[0, 0, 0, i];
            if (score < threshold) {
                continue;
            }
            ditects.Add(i, score);
        }

        return ditects;

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
