package com.example.llmback.entity.Lock;


import lombok.Data;

@Data
public class RedisLock {

    String key;
    String uniqueValue;

}
