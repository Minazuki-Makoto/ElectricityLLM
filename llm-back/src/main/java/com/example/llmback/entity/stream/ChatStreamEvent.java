package com.example.llmback.entity.stream;

import lombok.Data;

import java.math.BigInteger;

@Data
public class ChatStreamEvent {
    private String streamId;
    private String taskId;
    private Integer sequence;
    private String eventType;
    private String content;
    private BigInteger userId;
    private BigInteger sessionId;
    private String errorCode;
    private String errorMessage;
}
