package com.example.llmback.entity;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigInteger;
@Data
@NoArgsConstructor
@AllArgsConstructor
public class PlotInfo {

    private BigInteger plotId;
    private BigInteger userId;
    private BigInteger sessionId;
    private BigInteger chatId;
    private String artifactType;
    private String bucketName;
    private String objectKey;
    private String etag;
}
