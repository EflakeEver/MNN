package com.taobao.android.mnndemo;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Build;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.taobao.android.mnn.MNNForwardType;
import com.taobao.android.mnn.MNNImageProcess;
import com.taobao.android.mnn.MNNNetInstance;
import com.taobao.android.utils.Common;
import com.taobao.android.utils.TxtFileReader;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;

public class KcDemoActivity extends AppCompatActivity {

    private static final String TAG = KcDemoActivity.class.getSimpleName();
    private TextView mTv;
    private Button mBtn;
    private final String KcModelFileName = "Ksyun/resnet50_sqsh_v2001.onnx.mnn";
    private String mKcModelPath;
    private MNNNetInstance mNetInstance;
    private MNNNetInstance.Config mConfig;
    private MNNNetInstance.Session mSession;
    private MNNNetInstance.Session.Tensor mInputTensor;
    private MNNImageProcess.Config dataConfig;
    private Matrix imgData;
    private final int InputWidth = 224;
    private final int InputHeight = 224;
    public static final int REQUEST = 1;
    public static String[] PERMISSIONS = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_kc_demo);
        initWidget();
        prepareModels();
    }

    private void initWidget() {
        mTv = findViewById(R.id.tv_status);
        mBtn = findViewById(R.id.btn_load);
        mBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    requestPermission(KcDemoActivity.this);
                    loadModel();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    private void requestPermission(Activity activity) {

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (ContextCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
                    || ContextCompat.checkSelfPermission(activity, Manifest.permission.READ_SMS) != PackageManager.PERMISSION_GRANTED
            ) {
                // 检查权限状态
                if (ActivityCompat.shouldShowRequestPermissionRationale(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                    //  用户彻底拒绝授予权限
                } else {
                    //  用户未彻底拒绝授予权限
                    ActivityCompat.requestPermissions(activity, PERMISSIONS, REQUEST);
                }
            } else {
                Toast.makeText(activity, "已经授权 ： " + Manifest.permission.WRITE_EXTERNAL_STORAGE, Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 1) {
            for (int i = 0; i < grantResults.length; i++) {
                if (grantResults[i] == PackageManager.PERMISSION_GRANTED) {
                    // 申请成功
                    Toast.makeText(KcDemoActivity.this, "申请成功 ： " + PERMISSIONS[i], Toast.LENGTH_SHORT).show();
                } else {
                    // 申请失败
                    Toast.makeText(KcDemoActivity.this, "申请失败 ： " + PERMISSIONS[i], Toast.LENGTH_SHORT).show();
                }
            }
        }
    }


    private void prepareModels() {
//        mKcModelPath = getCacheDir() + "mobilenet_v1.caffe.mnn";
        mKcModelPath = getCacheDir() + "resnet50_sqsh_v2001.onnx.mnn";
        try {
            Common.copyAssetResource2File(getBaseContext(), KcModelFileName, mKcModelPath);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    private void loadModel() throws Exception {
        // create net instance
        mNetInstance = MNNNetInstance.createFromFile(mKcModelPath);
        // create session with config
        mConfig = new MNNNetInstance.Config();
        mConfig.numThread = 4;// set threads
        mConfig.forwardType = MNNForwardType.FORWARD_CPU.type;// set CPU/GPU
        // mConfig.saveTensors;
        mSession = mNetInstance.createSession(mConfig);

        // get input tensor
        mInputTensor = mSession.getInput(null);
//        String image_path = Environment.getExternalStorageDirectory() + File.separator + "1.jpg";
        String image_path = Environment.getExternalStorageDirectory() + File.separator + "1.bmp";
        if (!new File(image_path).exists()) {
            throw new Exception("image file is not exists!");
        }
        FileInputStream fis = new FileInputStream(image_path);
        Bitmap bmp = BitmapFactory.decodeStream(fis);
        Log.d("eflake","saveBitmap origin width = "+bmp.getWidth()+"height = "+ bmp.getHeight());
        File file = new File(Environment.getExternalStorageDirectory() + File.separator + "dest1.bmp");
        saveBitmap(bmp,file);
        dataConfig = new MNNImageProcess.Config();
//        dataConfig.mean = new float[]{103.94f, 116.78f, 123.68f};
//        dataConfig.normal = new float[]{0.017f, 0.017f, 0.017f};
        //图像的均值
        dataConfig.mean = new float[]{103.939f, 116.779f, 123.68f};
        //图像的标准差
        dataConfig.normal = new float[]{0.017429193899782133f, 0.01750700280112045f, 0.01712475383166367f};
        //预处理后的模型输入图片通道
        dataConfig.dest = MNNImageProcess.Format.BGR;
        imgData = new Matrix();
        imgData.postScale(InputWidth / (float) bmp.getWidth(), InputHeight / (float) bmp.getHeight());
//        imgData.postScale(1.0f, 1.0f);
        imgData.invert(imgData);
        MNNImageProcess.convertBitmap(bmp, mInputTensor, dataConfig, imgData);
        mInputTensor.getFloatData()
        File file2 = new File(Environment.getExternalStorageDirectory() + File.separator + "dest2.bmp");
        saveBitmap(bmp,file2);
        //bmp 224x224
        try {
            mSession.run();
        } catch (Exception e) {
            throw new Exception("predict image fail! log:" + e);
        }
        MNNNetInstance.Session.Tensor output = mSession.getOutput(null);
        float[] result = output.getFloatData();
        Log.d(TAG, Arrays.toString(result));
    }

    public boolean saveBitmap(Bitmap bitmap, File file) {
        if (bitmap == null)
            return false;
        FileOutputStream fos = null;
        try {
            Log.d("eflake","saveBitmap width = "+bitmap.getWidth()+"height = "+ bitmap.getHeight());
            fos = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
            fos.flush();
            return true;
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return false;
    }
}