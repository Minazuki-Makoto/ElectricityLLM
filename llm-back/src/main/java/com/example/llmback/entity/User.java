package com.example.llmback.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigInteger;
import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class User {

    private BigInteger userId;
    private String nickName;
    private String password;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;

}
