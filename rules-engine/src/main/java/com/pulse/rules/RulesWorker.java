package com.pulse.rules;

import com.pulse.common.dto.LogEvent;
import com.pulse.common.dto.RuleHit;
import org.springframework.stereotype.Component;

import java.util.ArrayList;

@Component
public class RulesWorker {

  public LogEvent applyRules(LogEvent ev) {
    if (ev.getRuleHits() == null) ev.setRuleHits(new ArrayList<>());

    String msg = ev.getMessage() == null ? "" : ev.getMessage().toLowerCase();

    // Example rule: keyword "login failed"
    if (msg.contains("login failed")) {
      RuleHit rh = new RuleHit();
      rh.setCode("AUTH_FAIL_KEYWORD");
      rh.setReason("Detected login failed phrase");
      rh.setWeight(0.45);
      ev.getRuleHits().add(rh);
    }

    return ev;
  }
}