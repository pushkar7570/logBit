package com.pulse.agg;

import com.pulse.common.adapter.InMemoryBus;
import com.pulse.common.dto.LogEvent;
import com.pulse.common.port.MessageBus;
import org.springframework.context.annotation.*;

@Configuration
public class AggConfig {
@Bean public MessageBus<LogEvent> logBus() { return new InMemoryBus<>(); }
}