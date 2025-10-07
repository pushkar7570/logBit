package com.pulse.common.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.time.Instant;
import java.util.*;

@JsonIgnoreProperties(ignoreUnknown = true)
public class LogEvent {
    private String id;
    private Instant ts;
    private String source;
    private String host;
    private String app;
    private String severityRaw;
    private String message;

    private Map<String, Object> features = new HashMap<>();
    private Map<String, String> tags = new HashMap<>();
    private Map<String, String> entities = new HashMap<>();
    private List<RuleHit> ruleHits = new ArrayList<>();

    public LogEvent() {
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public Instant getTs() {
        return ts;
    }

    public void setTs(Instant ts) {
        this.ts = ts;
    }

    public String getSource() {
        return source;
    }

    public void setSource(String source) {
        this.source = source;
    }

    public String getHost() {
        return host;
    }

    public void setHost(String host) {
        this.host = host;
    }

    public String getApp() {
        return app;
    }

    public void setApp(String app) {
        this.app = app;
    }

    public String getSeverityRaw() {
        return severityRaw;
    }

    public void setSeverityRaw(String severityRaw) {
        this.severityRaw = severityRaw;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public Map<String, Object> getFeatures() {
        return features;
    }

    public void setFeatures(Map<String, Object> features) {
        this.features = features;
    }

    public Map<String, String> getTags() {
        return tags;
    }

    public void setTags(Map<String, String> tags) {
        this.tags = tags;
    }

    public Map<String, String> getEntities() {
        return entities;
    }

    public void setEntities(Map<String, String> entities) {
        this.entities = entities;
    }

    public List<RuleHit> getRuleHits() {
        return ruleHits;
    }

    public void setRuleHits(List<RuleHit> ruleHits) {
        this.ruleHits = ruleHits;
    }
}
