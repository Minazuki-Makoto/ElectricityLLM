package com.example.llmback.entity;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigInteger;
import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class SessionChat {

    private BigInteger chatId;
    private BigInteger sessionId;
    private BigInteger userId;
    private String question;
    private String answer;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;

}
