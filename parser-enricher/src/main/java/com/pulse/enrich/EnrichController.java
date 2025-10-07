package com.pulse.enrich;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.pulse.common.dto.LogEvent;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
public class EnrichController {

    private final ObjectMapper om;

    public EnrichController(ObjectMapper om) {
        this.om = om;
    }

    @PostMapping("/v1/enrich")
    public LogEvent enrich(@RequestBody Map<String, Object> body) {
        // Convert the incoming JSON map into our shared DTO
        LogEvent ev = om.convertValue(body, LogEvent.class);

        if (ev.getFeatures() == null)
            ev.setFeatures(new HashMap<>());
        if (ev.getTags() == null)
            ev.setTags(new HashMap<>());
        if (ev.getEntities() == null)
            ev.setEntities(new HashMap<>());

        // (minimal enrichment – keep it simple for now)
        ev.getFeatures().putIfAbsent("enriched", true);

        return ev;
    }
}
