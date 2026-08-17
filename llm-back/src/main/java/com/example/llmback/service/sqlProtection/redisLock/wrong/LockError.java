package com.example.llmback.service.sqlProtection.redisLock.wrong;

public class LockError extends Exception{

    public LockError(String message){
        super(message);
    }
}
