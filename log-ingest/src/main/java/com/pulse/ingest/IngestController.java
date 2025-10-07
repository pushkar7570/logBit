package com.pulse.ingest;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.pulse.common.dto.LogEvent;
import com.pulse.common.port.MessageBus;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

// import java.nio.charset.StandardCharsets;
// import java.util.Arrays;

@RestController
@RequestMapping("/v1")
@RequiredArgsConstructor
public class IngestController {

    private final MessageBus<LogEvent> bus;
    private final ObjectMapper om = new ObjectMapper();

    /**
     * NDJSON endpoint.
     * Accept a single String body (WebFlux will bind it) to avoid any operator
     * confusion.
     * Split by newline, parse each line, publish to the bus, and return the IDs.
     */
    @PostMapping(value = "/logs", consumes = MediaType.APPLICATION_NDJSON_VALUE)
    public Flux<String> ingestNdjson(@RequestBody Flux<String> body) {
        return body
                .flatMap(s -> Flux.fromArray(s.split("\n")))
                .map(String::trim)
                .filter(line -> !line.isEmpty())
                .map(line -> {
                    try {
                        return om.readValue(line, LogEvent.class);
                    } catch (Exception e) {
                        throw new RuntimeException("Invalid NDJSON line: " + line, e);
                    }
                })
                .doOnNext(ev -> bus.publish("logs.raw", ev))
                .map(LogEvent::getId);
    }

    // Single JSON object fallback
    @PostMapping(value = "/logs", consumes = MediaType.APPLICATION_JSON_VALUE)
    public Mono<String> ingestJson(@RequestBody Mono<LogEvent> evMono) {
        return evMono.map(ev -> {
            bus.publish("logs.raw", ev);
            return ev.getId();
        });
    }
}
