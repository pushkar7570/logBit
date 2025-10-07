package com.pulse.enrich;


import com.pulse.common.adapter.InMemoryBus;
import com.pulse.common.dto.LogEvent;
import com.pulse.common.port.MessageBus;
import org.springframework.context.annotation.*;


@Configuration
public class EnrichConfig {
@Bean public MessageBus<LogEvent> logBus() { return new InMemoryBus<>(); }
}