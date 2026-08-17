package com.example.llmback.service.sqlProtection.redisLock.wrong;

public class ParametersError extends Exception{

    public ParametersError(String message){
        super(message);
    }
}
