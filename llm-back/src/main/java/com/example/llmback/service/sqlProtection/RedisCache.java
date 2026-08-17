package com.example.llmback.service.sqlProtection;

import com.example.llmback.entity.Lock.RedisLock;
import com.example.llmback.entity.SessionChat;
import com.example.llmback.entity.User;
import com.example.llmback.entity.UserSession;
import com.example.llmback.entity.response.Response;
import com.example.llmback.service.mapperService.MapperService;
import com.example.llmback.service.rocketService.MessageProducer;
import com.example.llmback.service.sqlProtection.redisLock.RedisDistributedLock;
import com.example.llmback.service.sqlProtection.redisLock.wrong.LockError;
import org.springframework.data.redis.core.script.DefaultRedisScript;
import tools.jackson.databind.ObjectMapper;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.math.BigInteger;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.*;
import java.util.function.Supplier;


@Service
public class RedisCache {

    private static final Logger log = LoggerFactory.getLogger(RedisCache.class);
    private static final Duration CACHE_TTL = Duration.ofDays(15);

    private final static DefaultRedisScript<Long> APPEND_LIST_SCRIPT;

    static {
        APPEND_LIST_SCRIPT = new DefaultRedisScript<>();
        APPEND_LIST_SCRIPT.setScriptText("""
                if redis.call('EXISTS', KEYS[1]) == 0 then
                    return 0
                end

                redis.call('RPUSH', KEYS[1], ARGV[1])
                redis.call('PEXPIRE', KEYS[1], ARGV[2])
                return 1
                """);

        APPEND_LIST_SCRIPT.setResultType(Long.class);
    }

    private final static DefaultRedisScript<Long> ADD_ZSET_SCRIPT;

    static {
        ADD_ZSET_SCRIPT = new DefaultRedisScript<>();
        ADD_ZSET_SCRIPT.setScriptText("""
                if redis.call('EXISTS', KEYS[1]) == 0 then
                    return 0
                end
                redis.call('ZADD', KEYS[1], ARGV[1],ARGV[2])
                redis.call('PEXPIRE', KEYS[1], ARGV[3])
                return 1
                """);
        ADD_ZSET_SCRIPT.setResultType(Long.class);
    }

    private final StringRedisTemplate redisTemplate;
    private final MapperService mapperService;
    private final ObjectMapper objectMapper;
    private final RedisDistributedLock redisDistributedLock;
    private final RedisLimiting redisLimiting;
    private final MessageProducer messageProducer;

    public RedisCache(StringRedisTemplate redisTemplate,
                      MapperService mapperService,
                      ObjectMapper objectMapper,
                      RedisDistributedLock redisDistributedLock,
                      RedisLimiting redisLimiting,
                      MessageProducer messageProducer
                      ) {
        this.redisTemplate = redisTemplate;
        this.mapperService = mapperService;
        this.objectMapper = objectMapper;
        this.redisDistributedLock = redisDistributedLock;
        this.redisLimiting = redisLimiting;
        this.messageProducer = messageProducer;
    }

    private String generateUserKey(String token){
        return "login:token:" + token;
    }
    private String generateZZSetKey(BigInteger userId){
        return "session:zset:user:" + userId;
    }
    private String generateListKey(BigInteger userId,
                                   BigInteger sessionId){
        return "chat:list:user:" + userId + ":session:" + sessionId;
    }

    public void addSessionToCache(UserSession session) {
        if (session == null || session.getUserId() == null) {
            throw new IllegalArgumentException("session和userId不能为空");
        }

        String sessionKey = generateZZSetKey(session.getUserId());
        String sessionString = convertJson(session);
        double score = calculateScore(session);

        redisTemplate.execute(
                ADD_ZSET_SCRIPT,
                List.of(sessionKey),
                String.valueOf(score),
                sessionString,
                String.valueOf(CACHE_TTL.toMillis())
        );
    }

    public void evictChats(BigInteger userId, BigInteger sessionId) {
        if (userId != null && sessionId != null) {
            redisTemplate.delete(generateListKey(userId, sessionId));
        }
    }

    public void upsertChat(SessionChat chat) {
        if (chat == null
                || chat.getChatId() == null
                || chat.getUserId() == null
                || chat.getSessionId() == null) {
            return;
        }

        String listKey = generateListKey(chat.getUserId(), chat.getSessionId());
        if (!Boolean.TRUE.equals(redisTemplate.hasKey(listKey))) {
            return;
        }

        String lockKey = redisDistributedLock.generateChatLockKey(
                chat.getUserId(),
                chat.getSessionId()
        );
        RedisLock lock = null;

        try {
            Optional<RedisLock> optionalLock = redisDistributedLock.tryLock(
                    lockKey,
                    Duration.ofSeconds(5)
            );
            if (optionalLock.isEmpty()) {
                redisTemplate.delete(listKey);
                return;
            }
            lock = optionalLock.get();

            List<String> cachedChats = redisTemplate.opsForList().range(listKey, 0, -1);
            if (cachedChats == null || cachedChats.isEmpty()) {
                return;
            }

            String chatJson = objectMapper.writeValueAsString(chat);
            for (int index = 0; index < cachedChats.size(); index++) {
                SessionChat cachedChat = objectMapper.readValue(
                        cachedChats.get(index),
                        SessionChat.class
                );
                if (chat.getChatId().equals(cachedChat.getChatId())) {
                    redisTemplate.opsForList().set(listKey, index, chatJson);
                    redisTemplate.expire(listKey, CACHE_TTL);
                    return;
                }
            }

            redisTemplate.execute(
                    APPEND_LIST_SCRIPT,
                    List.of(listKey),
                    chatJson,
                    String.valueOf(CACHE_TTL.toMillis())
            );
        } catch (Exception exception) {
            log.warn(
                    "更新聊天列表缓存失败，删除缓存以保证一致性 userId={}, sessionId={}, chatId={}",
                    chat.getUserId(),
                    chat.getSessionId(),
                    chat.getChatId(),
                    exception
            );
            redisTemplate.delete(listKey);
        } finally {
            if (lock != null) {
                try {
                    redisDistributedLock.removeLock(lock);
                } catch (Exception exception) {
                    log.warn("释放聊天列表缓存锁失败 lockKey={}", lockKey, exception);
                }
            }
        }
    }
    private String generatePlotKey(BigInteger userId,
                                   BigInteger sessionId,
                                   BigInteger chatId){
        return "plot:user:"+userId+"session:"+sessionId + "chat:"+chatId;
    }

    public Response<String> firstLogin(
            String name,
            String password,
            String ip
    ) throws Exception{

        String limitKey = redisLimiting.generateLimitingUserKey(ip);
        boolean allowed = redisLimiting.limitingRequest(
                limitKey,
                Duration.ofSeconds(60),
                10
        );

        if (!allowed){
            return new Response<String>(
                    "429",
                    "请求过于频繁",
                    null
            );
        }

        User user = mapperService.getUser(
                name,
                password
        );

        if (user == null){
            return new Response<String>(
                    "400",
                    "用户名信息错误或者密码错误，请检查",
                    null
            );
        }

        String newToken = generateToken(
                name
        );

        BigInteger newUserId = user.getUserId();

        redisTemplate.opsForValue().set(
                generateUserKey(newToken),
                newUserId.toString(),
                Duration.ofDays(15)
        );

        return new Response<String>(
                "200",
                "登录成功",
                newToken
        );
    }

    public Response<String> register(
            String name,
            String password,
            String ip
    ) throws Exception {
        if (name == null || name.isBlank()) {
            return new Response<>("400", "用户名不能为空", null);
        }
        if (password == null || password.isBlank()) {
            return new Response<>("400", "密码不能为空", null);
        }

        String normalizedName = name.strip();
        if (normalizedName.length() > 100) {
            return new Response<>("400", "用户名不能超过100个字符", null);
        }
        if (password.length() > 255) {
            return new Response<>("400", "密码不能超过255个字符", null);
        }
        if (mapperService.userNameExists(normalizedName)) {
            return new Response<>("409", "用户名已存在", null);
        }

        int affectedRows = mapperService.addUser(normalizedName, password);
        if (affectedRows != 1) {
            return new Response<>("500", "注册失败", null);
        }

        return firstLogin(normalizedName, password, ip);
    }

    public Response<BigInteger> validateToken (
            String token
    ) throws Exception{


        if (token == null || token.isBlank()){
            return new Response<BigInteger>(
                    "401",
                    "会话token失效，token无效",
                    null
            );
        }

        String key = generateUserKey(token);
        String userId = redisTemplate.opsForValue().get(key);

        if  (userId == null){
            return new Response<BigInteger>(
                    "401",
                    "会话token失效，请重新登录",
                    null
            );
        }

        return new Response<BigInteger>(
                "200",
                "登录成功",
                new BigInteger(userId)
        );
    }


    public List<UserSession> getAllSession(
            String token
    ) throws Exception{
        BigInteger userId = getUserIdByToken(token);
        String zSetKey = generateZZSetKey(userId);
        List<UserSession> cachedSessions = readSessionsFromCache(zSetKey);

        if (cachedSessions != null){
            return cachedSessions;
        }

        String lockKey = redisDistributedLock.generateSessionsLockKey(userId);
        Optional<RedisLock> optional = redisDistributedLock.tryLock(lockKey,Duration.ofSeconds(3));

        RedisLock redisLock = optional.orElseThrow(
                new Supplier<LockError>() {
                    @Override
                    public LockError get() {
                        return new LockError("");
                    }
                }
        );

        try{
            cachedSessions = readSessionsFromCache(zSetKey);
            if (cachedSessions != null){
                return cachedSessions;
            }

            List<UserSession> sessionList = mapperService.getAllSession(userId);

            sessionList.sort(
                    Comparator.comparing(
                            UserSession::getUpdateTime,
                            Comparator.nullsLast(
                                    Comparator.reverseOrder()
                            )
                    )
            );

            for (UserSession session:sessionList){
                String sessionJson = convertJson(session);
                double score = calculateScore(session);

                redisTemplate.opsForZSet().add(
                        zSetKey,
                        sessionJson,
                        score
                );
            }
            if (!sessionList.isEmpty()) {
                redisTemplate.expire(
                        zSetKey,
                        Duration.ofDays(15)
                );
            }
            return sessionList;
        }finally {
            redisDistributedLock.removeLock(redisLock);
        }
    }

    public List<SessionChat> getAllChat(
            String token,
            BigInteger sessionId
    ) throws Exception{
            BigInteger userId = getUserIdByToken(token);

            String lockKey = redisDistributedLock.generateChatLockKey(userId,sessionId);
            String limitingKey = redisLimiting.generateLimitingSessionKey(userId,sessionId);
            String listKey = generateListKey(userId,sessionId);

            boolean allowed = redisLimiting.limitingRequest(
                    limitingKey,
                    Duration.ofSeconds(10),
                    20
            );

            if (!allowed){
                throw new IllegalStateException("请求过于频繁，请稍后重试");
            }

            List<SessionChat> cachedChats = readChatsFromCache(listKey);
            if (cachedChats != null){
                return cachedChats;
            }

            Optional<RedisLock> option = redisDistributedLock.tryLock(lockKey,Duration.ofSeconds(3));
            RedisLock lock = option.orElseThrow(
                    new Supplier<LockError>() {
                        @Override
                        public LockError get() {
                            return new LockError("当前锁被占用");
                        }
                    }
            );

            try{
                cachedChats = readChatsFromCache(listKey);
                if (cachedChats != null){
                    return cachedChats;
                }

                List<SessionChat> sessionChats = mapperService.getAllChat(
                        userId,
                        sessionId
                );

                List<String> chatsString = convertList(sessionChats);
                if (!chatsString.isEmpty()) {
                    redisTemplate.opsForList().rightPushAll(listKey,chatsString);
                    redisTemplate.expire(
                            listKey,
                            Duration.ofDays(15)
                    );
                }

                return sessionChats;
            }finally {
                redisDistributedLock.removeLock(lock);
            }
    }

    public List<String> getAllPlotURLS(BigInteger userId,
                                   BigInteger sessionId,
                                   BigInteger chatId) throws Exception{
        String redisPlotKey = generatePlotKey(
                userId,
                sessionId,
                chatId
        );

        String plotLockKey = redisDistributedLock.generatePlotLockKey(
                userId,
                sessionId,
                chatId
        );
        List<String> URLS = redisTemplate.opsForList().range(redisPlotKey,0,-1);
        if (Objects.equals(URLS, List.of("NONE"))){
            // Remove legacy negative cache so a later database insert can be observed.
            redisTemplate.delete(redisPlotKey);
            URLS = List.of();
        }

        if (URLS != null && !URLS.isEmpty()){
            return URLS;
        }

        Optional<RedisLock> optional = redisDistributedLock.tryLock(
                plotLockKey,
                Duration.ofSeconds(3)
        );

        RedisLock lock = optional.orElseThrow(
                new Supplier<LockError>() {
                    @Override
                    public LockError get() {
                        return new LockError("当前锁被占用");
                    }
                }
        );

        try{
            List<String> URLSLoad =mapperService.getURLS(userId,sessionId,chatId);
            if (URLSLoad.isEmpty()){
                return List.of();
            }

            redisTemplate.opsForList().rightPushAll(redisPlotKey, URLSLoad);
            // Presigned URLs are presentation data, so cache them only briefly.
            redisTemplate.expire(redisPlotKey, Duration.ofMinutes(40));

            return URLSLoad;

        }finally {
            redisDistributedLock.removeLock(lock);
        }
    }
    
    private List<UserSession> readSessionsFromCache(String zSetKey){
        Set<String> sessions = redisTemplate.opsForZSet().reverseRange(
                zSetKey,
                0,
                -1
        );

        if (sessions == null || sessions.isEmpty()){
            return null;
        }

        List<UserSession> sessionList = new ArrayList<>();
        for (String session : sessions){
            try{
                sessionList.add(
                        objectMapper.readValue(
                                session,
                                UserSession.class
                        )
                );
            } catch (Exception e){
                redisTemplate.delete(zSetKey);
                return null;
            }
        }

        return sessionList;
    }

    private List<SessionChat> readChatsFromCache(String listKey){
        List<String> chatList = redisTemplate.opsForList().range(
                listKey,
                0,
                -1
        );

        if (chatList == null || chatList.isEmpty()){
            return null;
        }

        List<SessionChat> chats = new ArrayList<>();
        for (String chat : chatList){
            try{
                chats.add(
                        objectMapper.readValue(
                                chat,
                                SessionChat.class
                        )
                );
            } catch (Exception e){
                redisTemplate.delete(listKey);
                return null;
            }
        }

        return chats;
    }

    private List<String> convertList(List<SessionChat> sessionChats){
        List<String> stringList = new ArrayList<>();

        for (SessionChat sessionChat:sessionChats){

            String chatJson = objectMapper.writeValueAsString(sessionChat);
            stringList.add(chatJson);
        }
        return stringList;
    }

    public BigInteger getUserIdByToken(String token) throws Exception{
        if (token == null || token.isBlank()) {
            throw new IllegalStateException("未提供token");
        }

        String key = generateUserKey(token);
        String userId = redisTemplate.opsForValue().get(key);

        if (userId == null) {
            throw new IllegalStateException("token无效或已过期");
        }

        return new BigInteger(userId);
    }

    private String convertJson(UserSession userSession){
        if (userSession == null){
            throw new IllegalArgumentException("userSession不能为空");
        }

        return objectMapper.writeValueAsString(userSession);
    }

    private double calculateScore(UserSession userSession){
        LocalDateTime time = userSession.getUpdateTime();

        if (time == null) {
            time = userSession.getCreateTime();
        }

        if (time == null) {
            return System.currentTimeMillis();
        }

        return time.atZone(ZoneId.of("Asia/Shanghai")).
                toInstant().
                toEpochMilli();

    }

    private String generateToken(
            String name
    ){
        return "token:" + UUID.randomUUID();
    }

}
