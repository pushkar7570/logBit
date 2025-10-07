package com.pulse.rules;

import com.pulse.common.adapter.InMemoryBus;
import com.pulse.common.dto.LogEvent;
import com.pulse.common.port.MessageBus;
import org.springframework.context.annotation.*;

@Configuration
public class RulesConfig {
@Bean public MessageBus<LogEvent> logBus() { return new InMemoryBus<>(); }
}