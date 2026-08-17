package com.example.llmback.service.minioService;


import io.minio.GetPresignedObjectUrlArgs;


import io.minio.Http;
import io.minio.MinioClient;
import io.minio.errors.MinioException;
import org.springframework.stereotype.Service;

import java.util.concurrent.TimeUnit;

@Service
public class ParsePlotURL {

    private final MinioClient minioClient;

    public ParsePlotURL(MinioClient minioClient) {
        this.minioClient = minioClient;
    }

    public String generateURL(String bucketName,
                              String objectName){
        try{
            return minioClient.getPresignedObjectUrl(
                    GetPresignedObjectUrlArgs.builder().
                            method(Http.Method.GET).
                            bucket(bucketName).
                            object(objectName).
                            expiry(1, TimeUnit.HOURS).
                            build()
            );
        } catch (MinioException e) {
            throw new RuntimeException(
                    "生成 MinIO 图片访问地址失败，bucket="
                            + bucketName
                            + ", object="
                            + objectName,
                    e
            );
        }
    }
}
