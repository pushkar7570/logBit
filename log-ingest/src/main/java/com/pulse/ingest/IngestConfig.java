package com.pulse.ingest;


import com.pulse.common.adapter.InMemoryBus;
import com.pulse.common.dto.LogEvent;
import com.pulse.common.port.MessageBus;
import org.springframework.context.annotation.*;


@Configuration
public class IngestConfig {
@Bean public MessageBus<LogEvent> logBus() { return new InMemoryBus<>(); }
}