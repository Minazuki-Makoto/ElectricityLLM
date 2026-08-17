package com.example.llmback.service.sqlProtection.redisLock;


import com.example.llmback.entity.Lock.RedisLock;
import com.example.llmback.service.sqlProtection.redisLock.wrong.LockError;
import com.example.llmback.service.sqlProtection.redisLock.wrong.ParametersError;
import com.example.llmback.service.sqlProtection.redisLock.wrong.RemoveLockError;
import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.script.DefaultRedisScript;
import org.springframework.stereotype.Component;

import java.math.BigInteger;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

@Slf4j
@Component
public class RedisDistributedLock {

    private final StringRedisTemplate stringRedisTemplate;

    private static final DefaultRedisScript<Long> UNLOCK_SCRIPT;

    static {
        UNLOCK_SCRIPT =new DefaultRedisScript<>();
        UNLOCK_SCRIPT.setScriptText( """
                if redis.call('get',KEYS[1])== ARGV[1] then
                    return redis.call('del',KEYS[1])
                else
                    return 0
                end
                """);
        UNLOCK_SCRIPT.setResultType(Long.class);
    }

    private static final DefaultRedisScript<Long> RENEW_SCRIPT;
    static{
        RENEW_SCRIPT = new DefaultRedisScript<>();
        RENEW_SCRIPT.setScriptText("""
                if redis.call('get',KEYS[1]) == ARGV[1] then
                    return redis.call('pexpire',KEYS[1],ARGV[2])
                else
                    return 0
                end
                """);

        RENEW_SCRIPT.setResultType(Long.class);
    }

    private final ConcurrentHashMap<String, ScheduledFuture<?>> watchDogTasks = new ConcurrentHashMap<>();

    private static final AtomicInteger WATCHDOG_THREAD_NUMBER = new AtomicInteger(0);

    private final ScheduledExecutorService watchDogExecutor =
            Executors.newScheduledThreadPool(
                    2,
                    runnable -> {
                        Thread thread = new Thread(
                                runnable,
                                "redis-lock-watchdog-" + WATCHDOG_THREAD_NUMBER.incrementAndGet()
                        );
                        thread.setDaemon(true);

                        return thread;
                    }
            );

    public RedisDistributedLock(StringRedisTemplate stringRedisTemplate) {
        this.stringRedisTemplate = stringRedisTemplate;
    }

    public Optional<RedisLock> tryLock(String lockKey, Duration ttl) throws Exception{
        if (ttl == null || ttl.isZero() || ttl.isNegative()){
            throw new ParametersError("ttl必须大于0");
        }
        if (lockKey == null || lockKey.isBlank()){
            throw new ParametersError("key字段不能为空");
        }

        RedisLock  redisLock = generateLock(lockKey);

        Boolean acquired = stringRedisTemplate.
                opsForValue().
                setIfAbsent(
                redisLock.getKey(),redisLock.getUniqueValue(),ttl
        );

        if(Boolean.TRUE.equals(acquired)){
            return Optional.of(redisLock);
        }

        return Optional.empty();

    }



    public void removeLock(RedisLock redisLock) throws Exception {
        if (redisLock == null){
            throw new LockError("当前单节点锁为空");
        }

        if (redisLock.getKey() == null || redisLock.getKey().isBlank()){
            throw new ParametersError("Lock中key字段不能为空");
        }

        cancelWatchDog(generateWatchDogTaskId(redisLock));

        Long result = stringRedisTemplate.execute(
                UNLOCK_SCRIPT,
                List.of(redisLock.getKey()),
                redisLock.getUniqueValue()
        );

        if (Long.valueOf(1L).equals(result)){
            return;
        }

        throw new RemoveLockError("删除锁失败");
    }


    public Optional<RedisLock> tryLockWithWatchDog(String lockKey,Duration ttl) throws Exception{

        if (lockKey == null || lockKey.isBlank()){
            throw new ParametersError("lock-key不能为空字段");
        }

        if (ttl == null || ttl.isZero() || ttl.isNegative()){
            throw new ParametersError("时间间隔不能为0");
        }

        RedisLock redisLock = generateLock(lockKey);
        Boolean acquired = stringRedisTemplate.opsForValue().setIfAbsent(
                lockKey,
                redisLock.getUniqueValue(),
                ttl
        );

        if (!Boolean.TRUE.equals(acquired)){
            return Optional.empty();
        }

        startWatchDog(redisLock,ttl);

        return Optional.of(redisLock);
    }



    private void startWatchDog(RedisLock redisLock,
                               Duration ttl) {
        long ttlMillis = ttl.toMillis();

        long renewEndTime = Math.max(100, ttlMillis / 3);

        String taskId = generateWatchDogTaskId(redisLock);

        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                try {
                    Long result = stringRedisTemplate.execute(
                            RENEW_SCRIPT,
                            List.of(redisLock.getKey()),
                            redisLock.getUniqueValue(),
                            String.valueOf(ttlMillis)
                    );

                    if (!Long.valueOf(1L).equals(result)) {
                        log.error(
                                "Redis锁已不存在或所有权已转移，停止看门狗，key={}",
                                redisLock.getKey()
                        );

                        cancelWatchDog(taskId);
                    }
                } catch (Exception e) {
                    log.error(
                            "Redis锁续期异常，key={}",
                            redisLock.getKey(),
                            e
                    );
                }
            }
        };

        ScheduledFuture<?> future = watchDogExecutor.scheduleWithFixedDelay(
                runnable,
                renewEndTime,
                renewEndTime,
                TimeUnit.MILLISECONDS
        );

        ScheduledFuture<?> oldFuture = watchDogTasks.putIfAbsent(
                taskId,
                future
        );

        if (oldFuture != null) {
            future.cancel(false);
        }

    }


    private void cancelWatchDog(String taskId){
        ScheduledFuture<?> future =
                watchDogTasks.remove(taskId);

        if (future != null) {
            /*
             * false：
             * 如果当前这一轮任务正在执行，不强制中断；
             * 只取消后续周期执行。
             */
            future.cancel(false);
        }
    }

    private String generateWatchDogTaskId(RedisLock redisLock){

        return redisLock.getKey()+":"+redisLock.getUniqueValue();
    }

    @PreDestroy
    public void shutdownWatchDog(){
        watchDogTasks.values().forEach(
                future -> future.cancel(false)
        );
        watchDogTasks.clear();
        watchDogExecutor.shutdown();
    }

    private RedisLock generateLock(String key) throws ParametersError{
        RedisLock redisLock = new RedisLock();

        if (key ==null){
            throw new ParametersError("key字段不能为空");
        }

        redisLock.setKey(key);

        String value = UUID.randomUUID().toString();
        redisLock.setUniqueValue(value);

        return redisLock;
    }

    /*根据前端传来的token查userId建立锁*/
    public String generateTokenLockKey(String token){
        return "llm-back:lock:token:"+token;
    }

    /*根据查询用户所有的sessions建立锁*/
    public String generateSessionsLockKey(BigInteger userId){
        return "llm-back:lock:sessions:"+userId;
    }

    /*根据查询用户某一个session里的全部chat建立锁*/
    public String generateChatLockKey(BigInteger userId,BigInteger sessionId){
        return "llm-back:lock:chat:"+userId+":"+sessionId;
    }

    public String generatePlotLockKey(BigInteger userId,BigInteger sessionId,BigInteger chatId){
        return "llm-back:lock:plot"+userId+":"+sessionId+":"+chatId;
    }

}
