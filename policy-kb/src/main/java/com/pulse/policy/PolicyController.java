package com.pulse.policy;

import com.pulse.common.dto.PolicySnippet;
import org.springframework.web.bind.annotation.*;

import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/v1/policies")
public class PolicyController {

  private final List<PolicySnippet> store = new CopyOnWriteArrayList<>();

  @GetMapping("/health")
  public Map<String, String> health() { return Map.of("status", "ok"); }

  // quick debug helpers
  @GetMapping("/_count")
  public Map<String,Object> count() { return Map.of("count", store.size()); }

  @GetMapping("/_dump")
  public List<PolicySnippet> dump() { return store; }

  @PostMapping("/upload")
  public Map<String, Object> upload(@RequestBody List<String> chunks) {
    if (chunks != null) {
      for (String t : chunks) {
        if (t == null) continue;
        PolicySnippet ps = new PolicySnippet();
        ps.setText(t);
        ps.setSimilarity(0.0);
        ps.setSectionRef("stub");
        store.add(ps);
      }
    }
    return Map.of("count", store.size());
  }

  @PostMapping("/uploadSnippets")
  public Map<String, Object> uploadSnippets(@RequestBody List<PolicySnippet> snippets) {
    if (snippets != null) {
      for (PolicySnippet s : snippets) {
        if (s == null) continue;
        if (s.getText() == null) s.setText("");
        if (s.getSectionRef() == null) s.setSectionRef("stub");
        store.add(s); // similarity defaults to 0.0 (primitive double)
      }
    }
    return Map.of("count", store.size());
  }

  @GetMapping("/search")
  public List<PolicySnippet> search(
    @RequestParam(name = "q", required = false) String q,                                
    @RequestParam(name = "k", defaultValue = "5") int k) {

    try {
      String query = q == null ? "" : q.trim();
      int topK = Math.max(0, k);
      if (store.isEmpty()) return List.of();
      if (query.isEmpty()) return store.stream().limit(topK).collect(Collectors.toList());

      return store.stream()
          .filter(Objects::nonNull)
          .map(s -> withScoreCopy(s, cosineLike(query, safe(s.getText()))))
          .sorted(Comparator.comparingDouble(PolicySnippet::getSimilarity).reversed())
          .limit(topK)
          .collect(Collectors.toList());
    } catch (Exception e) {
      e.printStackTrace(); // shows in the policy-kb terminal
      // return a safe “empty result” instead of 500 while debugging
      return List.of();
    }
  }

  private PolicySnippet withScoreCopy(PolicySnippet s, double score) {
    PolicySnippet out = new PolicySnippet();
    out.setText(safe(s.getText()));
    out.setSectionRef(safe(s.getSectionRef()));
    out.setSimilarity(score);
    return out;
  }

  private String safe(String v) { return v == null ? "" : v; }

  private double cosineLike(String a, String b) {
    Set<String> A = tokens(safe(a));
    Set<String> B = tokens(safe(b));
    if (A.isEmpty() || B.isEmpty()) return 0.0;
    long inter = A.stream().filter(B::contains).count();
    return inter / (double) Math.max(A.size(), B.size());
  }

  private Set<String> tokens(String s) {
    if (s.isBlank()) return Set.of();
    String[] parts = s.toLowerCase().split("\\W+");
    Set<String> out = new HashSet<>();
    for (String p : parts) if (!p.isBlank()) out.add(p);
    return out;
  }
}
