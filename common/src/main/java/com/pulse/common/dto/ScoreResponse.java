package com.pulse.common.dto;
import lombok.*;
import java.util.*;

@Data @Builder @NoArgsConstructor @AllArgsConstructor
public class ScoreResponse {
@Builder.Default private Map<String,Double> zeroShot = new HashMap<>();
private double anomaly;
private double policySimMax;
@Builder.Default private List<String> policyRefs = new ArrayList<>();
}