package com.example.llmback.entity.message;


import lombok.Data;

import java.math.BigInteger;

@Data
public class RocketMessage {

    private String taskId;
    private String topic;
    private String status;
    private String question;
    private BigInteger userId;
    private BigInteger sessionId;
    private BigInteger chatId;

    private String failureReason;
}
