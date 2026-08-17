package com.example.llmback.config;


import io.minio.MinioClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MinIOClientConfig {

    @Value("${minio.secret-key}")
    private String secretKey;

    @Value("${minio.access-key}")
    private String accessKey;

    @Value("${minio.endpoint}")
    private String endPoint;

    @Bean
    public MinioClient build(){

        return MinioClient.builder().
                endpoint(endPoint).
                credentials(accessKey,secretKey).
                build();
    }
}