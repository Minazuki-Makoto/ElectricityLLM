package com.example.llmback.service.sqlProtection.redisLock.wrong;

public class RemoveLockError extends Exception{

    public RemoveLockError(String message){
        super(message);
    }
}
