package com.example.llmback.mapper;


import com.example.llmback.entity.User;
import org.apache.ibatis.annotations.*;

import java.math.BigInteger;

@Mapper
public interface UserMapper {

    @Select("""
SELECT id AS userId,
       nick_name AS nickName,
       password ,
       create_time AS createTime,
       update_time AS updateTime
FROM llm_user_table
WHERE nick_name = #{name}
AND password = #{password}
""")
    User getUser(@Param("name") String name,
                 @Param("password") String password);

    @Select("""
SELECT COUNT(*)
FROM llm_user_table
WHERE nick_name = #{name}
""")
    int countByName(@Param("name") String name);

    @Insert("""
INSERT INTO llm_user_table (nick_name,password,create_time,update_time)
VALUES (
        #{name},
        #{password},
        Now(),
        Now()
)
""")
    int addUser(
            @Param("name") String name,
            @Param("password") String password
    );

    @Update("""
UPDATE llm_user_table
SET nick_name = #{name},
    update_time = Now()
WHERE id = #{userId} 
AND password = #{password}
""")
    int updateName(
            @Param("name") String name,
            @Param("userId") BigInteger userId,
            @Param("password") String password
    );

    @Delete("""
DELETE FROM llm_user_table
WHERE id = #{userId}
AND nick_name = #{name}
AND password = #{password}
""")
    int deleteUser(
            @Param("userId") BigInteger userId,
            @Param("name") String name,
            @Param("password") String password
    );



}
