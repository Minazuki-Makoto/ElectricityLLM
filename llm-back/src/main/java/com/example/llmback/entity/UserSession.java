package com.example.llmback.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigInteger;
import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class UserSession {

    private BigInteger sessionId;
    private BigInteger userId;
    private String sessionName;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;

}
