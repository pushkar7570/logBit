package com.pulse.enrich;


import com.pulse.common.dto.LogEvent;
import com.pulse.common.port.MessageBus;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;


import java.util.regex.Pattern;


@Component @RequiredArgsConstructor
public class EnrichWorker {
private final MessageBus<LogEvent> bus;
private static final Pattern EMAIL = Pattern.compile("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}");


@PostConstruct void init() {
bus.subscribe("logs.raw", this::process);
}


void process(LogEvent ev) {
// naive PII masking (emails)
ev.setMessage(EMAIL.matcher(ev.getMessage()).replaceAll("***@***"));
// simple features
ev.getFeatures().putIfAbsent("msg_len", ev.getMessage().length());
ev.getTags().putIfAbsent("env", ev.getTags().getOrDefault("env", "prod"));
bus.publish("logs.enriched", ev);
}
}