package com.example.llmback.dto.response;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.math.BigInteger;

@Data
@AllArgsConstructor
public class ChatResponse {
    private BigInteger sessionId;
    private BigInteger chatId;
    private String taskId;
    private boolean firstChat;
}
