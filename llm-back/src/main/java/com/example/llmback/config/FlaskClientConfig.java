package com.example.llmback.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

import java.net.http.HttpClient;
import java.time.Duration;

@Configuration
public class FlaskClientConfig {
    @Value("${flask.url}")
    private String url;

    @Value("${flask.connect-timeout-seconds:10}")
    private long connectTimeoutSeconds;

    @Value("${flask.read-timeout-seconds:330}")
    private long readTimeoutSeconds;


    @Bean
    public RestClient FlaskClient(){
        HttpClient httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(connectTimeoutSeconds))
                .build();
        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(httpClient);
        requestFactory.setReadTimeout(Duration.ofSeconds(readTimeoutSeconds));

        return RestClient.builder().
                baseUrl(url).
                requestFactory(requestFactory).
                build();
    }
}
