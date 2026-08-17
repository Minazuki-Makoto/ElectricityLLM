package com.example.llmback.entity.response;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class Response<T> {
    String status;
    String message;
    T data;
}
