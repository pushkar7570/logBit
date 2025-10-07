package com.pulse.common.dto;
import lombok.*;
import java.util.*;

@Data @Builder @NoArgsConstructor @AllArgsConstructor
public class ScoreRequest {
private String alertText;
@Builder.Default private List<PolicySnippet> policySnippets = new ArrayList<>();
@Builder.Default private Map<String,Object> features = new HashMap<>();
private boolean ruleHit; // aggregated
}