package org.example.runner;

import ai.onnxruntime.*;

import java.nio.LongBuffer;
import java.util.Map;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {

     static void main(String[] args) throws OrtException {
        // 1. Ortamı ve oturumu başlat
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        OrtSession session = env.createSession("C:\\Users\\Burhan Cerit\\PycharmProjects\\PythonProject\\shakespeare_model.onnx", opts);

        // 2. Girdi verisini oluştur (1 satır, 64 token)
        long[] inputData = new long[64];
        OnnxTensor tensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputData), new long[]{1, 64});

        // 3. Tahmin yap
        Map<String, OnnxTensor> inputs = Map.of("input_tokens", tensor);
        OrtSession.Result results = session.run(inputs);

        // 4. Sonucu al
        float[][][] logits = (float[][][]) results.get(0).getValue();
        System.out.println("Çıktı alındı! Boyut: " + logits.length);

        session.close();
        env.close();
    }
}
