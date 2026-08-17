package com.example.llmback.service;

import com.example.llmback.entity.PlotInfo;
import com.example.llmback.entity.SessionChat;
import com.example.llmback.entity.message.RocketMessage;
import com.example.llmback.service.mapperService.MapperService;
import com.example.llmback.service.sqlProtection.RedisCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;
import org.springframework.web.client.RestClient;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class FullService {

    private static final Logger log = LoggerFactory.getLogger(FullService.class);

    private final RestClient flaskClient;
    private final MapperService mapperService;
    private final RedisCache redisCache;

    public FullService(RestClient flaskClient,
                       MapperService mapperService,
                       RedisCache redisCache) {
        this.flaskClient = flaskClient;
        this.mapperService = mapperService;
        this.redisCache = redisCache;
    }

    @Transactional
    public void processMessage(RocketMessage message) throws Exception {
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("query", message.getQuestion());
        requestBody.put("user_id", message.getUserId());
        requestBody.put("session_id", message.getSessionId());
        requestBody.put("task_id", message.getTaskId());
        if (message.getChatId() != null) {
            requestBody.put("chat_id", message.getChatId());
        }

        Map<?, ?> response = flaskClient.post()
                .uri("/ai")
                .body(requestBody)
                .retrieve()
                .body(Map.class);

        String answer = extractFinalAnswer(response);
        List<PlotInfo> artifacts = extractArtifacts(response);

        SessionChat chat = mapperService.completeChat(message.getTaskId(), answer);
        BigInteger chatId = chat.getChatId();

        mapperService.insertPlots(chat.getUserId(), chat.getSessionId(), chatId, artifacts);
        updateChatCacheAfterCommit(chat);
    }

    private void updateChatCacheAfterCommit(SessionChat chat) {
        if (!TransactionSynchronizationManager.isSynchronizationActive()) {
            redisCache.upsertChat(chat);
            indexCompletedChatMemorySafely(chat);
            return;
        }

        TransactionSynchronizationManager.registerSynchronization(
                new TransactionSynchronization() {
                    @Override
                    public void afterCommit() {
                        redisCache.upsertChat(chat);
                        indexCompletedChatMemorySafely(chat);
                    }
                }
        );
    }

    private void indexCompletedChatMemorySafely(SessionChat chat) {
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("chat_id", chat.getChatId());
        requestBody.put("user_id", chat.getUserId());
        requestBody.put("session_id", chat.getSessionId());

        try {
            flaskClient.post()
                    .uri("/memory/index")
                    .body(requestBody)
                    .retrieve()
                    .toBodilessEntity();
            log.info("Memory indexing completed, chatId={}", chat.getChatId());
        } catch (Exception exception) {
            log.error(
                    "Memory indexing failed after chat transaction committed, chatId={}",
                    chat.getChatId(),
                    exception
            );
        }
    }

    public void markFailed(String taskId, String errorMessage) {
        mapperService.markChatFailed(taskId, errorMessage);
    }

    private String extractFinalAnswer(Map<?, ?> response) {
        if (response == null || !"success".equals(response.get("status"))) {
            throw new IllegalStateException("Python service returned a failure response");
        }
        Object message = response.get("message");
        if (!(message instanceof Map<?, ?> messageMap)) {
            throw new IllegalStateException("Python response message has an invalid format");
        }
        Object finalAnswer = messageMap.get("final_answer");
        if (!(finalAnswer instanceof String answer) || answer.isBlank()) {
            throw new IllegalStateException("Python response final_answer is blank");
        }
        return answer;
    }

    private List<PlotInfo> extractArtifacts(Map<?,?> response){
        List<PlotInfo> plotInfos = new ArrayList<>();

        if (response == null || !"success".equals(response.get("status"))) {
            throw new IllegalStateException("Python service returned a failure response");
        }

        Object message = response.get("message");
        if(!(message instanceof Map<?,?> messageMap)){
            throw  new IllegalStateException("Python response message has an invalid format");
        }

        Object details = messageMap.get("details");

        if(!(details instanceof List<?> infos)){
            throw new IllegalStateException("Python response details have an invalid format");
        }

        for (Object info:infos){
            if (!(info instanceof Map<?,?>infoMap)){
                throw new IllegalStateException("Python response has an invalid format");
            }
            Object skill = infoMap.get("skill");
            if (!(skill instanceof String skillName) || !"plot".equals(skillName)) {
                continue;
            }

            Object artifacts = infoMap.get("artifacts");

            if(!(artifacts instanceof List<?>artifactsList)){
                throw new IllegalStateException("Plot results have some format error");
            }

            for (Object art :artifactsList){
                if(!(art instanceof Map<?,?> artMap)){
                    throw new IllegalStateException("plot results have some format error");
                }

                String bucketName = requiredString(artMap, "bucket_name");
                String objectKey = requiredString(artMap, "object_key");

                PlotInfo plotInfo = new PlotInfo();
                plotInfo.setArtifactType(optionalString(artMap, "type", "chart"));
                plotInfo.setBucketName(bucketName);
                plotInfo.setObjectKey(objectKey);
                plotInfo.setEtag(optionalString(artMap, "etag", null));
                plotInfos.add(plotInfo);
            }
        }

        return plotInfos;
    }

    private String requiredString(Map<?, ?> source, String key) {
        Object value = source.get(key);
        if (!(value instanceof String text) || text.isBlank()) {
            throw new IllegalStateException("Plot artifact " + key + " is blank");
        }
        return text;
    }

    private String optionalString(Map<?, ?> source, String key, String defaultValue) {
        Object value = source.get(key);
        return value instanceof String text && !text.isBlank() ? text : defaultValue;
    }
}
