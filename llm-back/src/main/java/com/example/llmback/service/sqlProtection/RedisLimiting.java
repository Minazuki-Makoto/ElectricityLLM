package com.example.llmback.service.sqlProtection;

import com.example.llmback.service.sqlProtection.redisLock.wrong.ParametersError;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.script.DefaultRedisScript;
import org.springframework.stereotype.Component;

import java.math.BigInteger;
import java.time.Duration;
import java.util.List;


@Component
public class RedisLimiting {

    private static final DefaultRedisScript<Long> REJECT_SCRIPT;

    static {
        REJECT_SCRIPT = new DefaultRedisScript<>();
        REJECT_SCRIPT.setScriptText("""
                local current = redis.call('INCR', KEYS[1])

                if current == 1 then
                    redis.call('PEXPIRE', KEYS[1], ARGV[1])
                    return 1
                end

                if current <= tonumber(ARGV[2]) then
                    return 1
                end

                return 0
                """);
        REJECT_SCRIPT.setResultType(Long.class);
    }

    private final StringRedisTemplate stringRedisTemplate;

    public RedisLimiting(StringRedisTemplate stringRedisTemplate) {
        this.stringRedisTemplate = stringRedisTemplate;
    }

    public boolean limitingRequest(String limitingKey,
                                Duration timeStep,
                                Integer times) throws Exception{

        if (limitingKey == null || limitingKey.isBlank()){
            throw new ParametersError("限流锁的key字段不能为空");
        }
        if (timeStep == null || timeStep.isZero() || timeStep.isNegative()){
            throw new ParametersError("限流时间窗口必须大于0");
        }
        if (times == null || times <= 0){
            throw new ParametersError("限流次数必须大于0");
        }

        Long result = stringRedisTemplate.execute(
                REJECT_SCRIPT,
                List.of(limitingKey),
                String.valueOf(timeStep.toMillis()),
                String.valueOf(times)
        );

        return Long.valueOf(1L).equals(result);
    }

    /*防止用户输完信息后连点*/
    public String generateLimitingUserKey(String ip){

        return "llm-back:limit:login:ip:"+ip;
    }

    /*防止用户短时间发送过多对某一session的全部会话请求*/
    public String generateLimitingSessionKey(BigInteger userId,
                                              BigInteger sessionId){

        return "llm-back:limit:chat:user:"
                + userId
                + ":session:"
                + sessionId;
    }

    public String generateChatSubmitKey(BigInteger userId) {
        if (userId == null) {
            throw new IllegalArgumentException("userId不能为空");
        }

        return "llm-back:limit:chat-submit:user:" + userId;
    }
}
