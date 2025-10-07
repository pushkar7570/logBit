package com.pulse.rules;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.pulse.common.dto.LogEvent;
import com.pulse.common.dto.RuleHit;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/v1/rules")
public class RulesController {

    private final ObjectMapper om;

    public RulesController(ObjectMapper om) {
        this.om = om;
    }

    @PostMapping("/evaluate")
    public LogEvent evaluate(@RequestBody Map<String, Object> body) {
        LogEvent ev = om.convertValue(body, LogEvent.class);

        // ensure maps/lists
        if (ev.getFeatures() == null)
            ev.setFeatures(new HashMap<>());
        if (ev.getTags() == null)
            ev.setTags(new HashMap<>());
        if (ev.getEntities() == null)
            ev.setEntities(new HashMap<>());
        if (ev.getRuleHits() == null)
            ev.setRuleHits(new java.util.ArrayList<>());

        // Simple rule: keyword match
        String msg = ev.getMessage() == null ? "" : ev.getMessage().toLowerCase();
        if (msg.contains("login failed")) {
            RuleHit rh = new RuleHit();
            rh.setCode("AUTH_FAIL_KEYWORD");
            rh.setReason("Detected login failed phrase");
            rh.setWeight(0.45); // used by aggregator weighting
            ev.getRuleHits().add(rh);
        }

        return ev;
    }
}
