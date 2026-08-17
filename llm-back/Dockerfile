FROM maven:3.9.11-eclipse-temurin-17 AS build

WORKDIR /build

COPY llm-back/settings.xml /root/.m2/settings.xml
COPY llm-back/pom.xml ./pom.xml

RUN mvn -B -DskipTests dependency:go-offline

COPY llm-back/src ./src
RUN mvn -B -DskipTests package

FROM eclipse-temurin:17-jre-jammy

ENV JAVA_TOOL_OPTIONS="-Xms128m -Xmx512m"

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=build /build/target/*.jar /app/app.jar

USER appuser
EXPOSE 8088

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD curl --fail --silent http://127.0.0.1:8088/internal/health || exit 1

ENTRYPOINT ["java", "-jar", "/app/app.jar"]
