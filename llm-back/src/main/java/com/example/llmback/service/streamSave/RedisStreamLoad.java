package com.example.llmback.service.streamSave;

import com.example.llmback.entity.stream.ChatStreamEvent;
import org.springframework.data.redis.connection.stream.MapRecord;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.connection.stream.ReadOffset;
import org.springframework.data.redis.connection.stream.StreamOffset;
import org.springframework.data.redis.connection.stream.StreamReadOptions;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import tools.jackson.databind.ObjectMapper;

import java.math.BigInteger;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@Service
public class RedisStreamLoad {
    private static final long INTERVAL_TIME_SECONDS = 2L;

    private final ObjectMapper objectMapper;
    private final StringRedisTemplate redisTemplate;
    private final long linkInterruptedTimeMillis;

    public RedisStreamLoad(
            ObjectMapper objectMapper,
            StringRedisTemplate redisTemplate,
            @Value("${app.sse.timeout-millis:600000}") long linkInterruptedTimeMillis
    ) {
        this.objectMapper = objectMapper;
        this.redisTemplate = redisTemplate;
        this.linkInterruptedTimeMillis = linkInterruptedTimeMillis;
    }

    private String generateKey(String taskId) {
        return "redis:stream:" + taskId;
    }

    public ChatStreamEvent transform2Stream(MapRecord<String, Object, Object> record) {
        Map<Object, Object> value = record.getValue();

        ChatStreamEvent event = new ChatStreamEvent();
        event.setStreamId(record.getId().getValue());
        event.setTaskId(String.valueOf(value.get("task_id")));
        event.setEventType(String.valueOf(value.get("event_type")));
        event.setSequence(Integer.parseInt(String.valueOf(value.get("sequence"))));
        event.setContent(String.valueOf(value.getOrDefault("content", "")));

        String metadataJson = String.valueOf(value.getOrDefault("metadata", "{}"));
        Map<?, ?> metadata = objectMapper.readValue(metadataJson, Map.class);

        Object userId = metadata.get("user_id");
        if (userId != null && !String.valueOf(userId).isBlank()) {
            event.setUserId(new BigInteger(String.valueOf(userId)));
        }

        Object sessionId = metadata.get("session_id");
        if (sessionId != null && !String.valueOf(sessionId).isBlank()) {
            event.setSessionId(new BigInteger(String.valueOf(sessionId)));
        }

        Object errorCode = metadata.get("error_code");
        if (errorCode != null) {
            event.setErrorCode(String.valueOf(errorCode));
        }

        Object errorMessage = metadata.get("error_message");
        if (errorMessage != null) {
            event.setErrorMessage(String.valueOf(errorMessage));
        }

        return event;
    }

    public List<ChatStreamEvent> readEvents(String taskId, String lastStreamId) {
        String redisStreamKey = generateKey(taskId);
        StreamReadOptions readOptions = StreamReadOptions.empty()
                .count(40)
                .block(Duration.ofSeconds(INTERVAL_TIME_SECONDS));
        StreamOffset<String> offset = StreamOffset.create(
                redisStreamKey,
                ReadOffset.from(lastStreamId)
        );

        List<MapRecord<String, Object, Object>> streamList =
                redisTemplate.opsForStream().read(readOptions, offset);

        if (streamList == null || streamList.isEmpty()) {
            return List.of();
        }

        return streamList.stream()
                .map(this::transform2Stream)
                .toList();
    }


    public void sendSSE(SseEmitter sseEmitter,
                        String taskId,
                        BigInteger userId){
        String lastStreamId = "0-0";
        long startTime = System.currentTimeMillis();

        try{
            while (System.currentTimeMillis()-startTime <= linkInterruptedTimeMillis){
                List<ChatStreamEvent> readStreams = readEvents(taskId,lastStreamId);

                for (ChatStreamEvent stream:readStreams){
                    lastStreamId = stream.getStreamId();
                    BigInteger streamUserId = stream.getUserId();

                    if (streamUserId == null || !streamUserId.equals(userId)){
                        throw new IllegalArgumentException("user_id字段有问题");
                    }

                    String errorCode = stream.getErrorCode();
                    String errorInfo = stream.getErrorMessage();
                    String type = stream.getEventType();

                    if(!(errorCode == null || errorInfo == null || Objects.equals(type, "error"))){
                        throw new RuntimeException();
                    }


                    sseEmitter.send(SseEmitter.event().
                            id(lastStreamId).
                            name(type).
                            data(stream, MediaType.APPLICATION_JSON).
                            build());

                    if (Objects.equals(type,"done")){
                        sseEmitter.complete();
                        return ;
                    }
                }
            }

            sseEmitter.send(SseEmitter.event().
                    name("time_out").
                    data("AI等待任务超时").
                    build());

            sseEmitter.complete();

        } catch (Exception e) {
            sseEmitter.completeWithError(e);
        }
    }
}
