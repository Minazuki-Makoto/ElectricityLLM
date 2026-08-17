package com.example.llmback.mapper;


import com.example.llmback.entity.PlotInfo;
import org.apache.ibatis.annotations.*;

import java.math.BigInteger;
import java.util.List;

@Mapper
public interface PlotMapper {

    @Insert("""
        INSERT INTO plot_info (
            user_id,
            session_id,
            chat_id,
            artifact_type,
            bucket_name,
            object_key,
            etag
        )
        VALUES (
            #{userId},
            #{sessionId},
            #{chatId},
            #{artifactType},
            #{bucketName},
            #{objectKey},
            #{etag}
        )
        """)
    @Options(
            useGeneratedKeys = true,
            keyProperty = "plotId",
            keyColumn = "id"
    )
    int insert(PlotInfo plotInfo);

    @Select("""
        SELECT
            id ,
            user_id ,
            session_id ,
            chat_id ,
            artifact_type,
            bucket_name,
            object_key,
            etag
        FROM plot_info
        WHERE user_id = #{userId}
          AND session_id = #{sessionId}
          AND chat_id = #{chatId}
        """)
    @Results({
            @Result(column = "id", property = "plotId"),
            @Result(column = "user_id", property = "userId"),
            @Result(column = "session_id", property = "sessionId"),
            @Result(column = "chat_id", property = "chatId"),
            @Result(column = "artifact_type", property = "artifactType"),
            @Result(column = "bucket_name", property = "bucketName"),
            @Result(column = "object_key", property = "objectKey"),
            @Result(column = "etag", property = "etag")
    })
    List<PlotInfo> selectByChatId(
            @Param("userId") BigInteger userId,
            @Param("sessionId") BigInteger sessionId,
            @Param("chatId") BigInteger chatId
    );

}
