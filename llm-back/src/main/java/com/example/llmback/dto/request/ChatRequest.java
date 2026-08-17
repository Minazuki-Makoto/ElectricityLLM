package com.example.llmback.dto.request;


import lombok.Data;

import java.math.BigInteger;

@Data
public class ChatRequest {
    private BigInteger sessionId;
    private String question;
}
