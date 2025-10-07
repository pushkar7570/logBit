package com.pulse.notify;

import com.pulse.common.dto.LogEvent;
import com.pulse.common.port.MessageBus;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class NotifyWorker {
    private final MessageBus<LogEvent> bus;

    @PostConstruct
    void init() {
        bus.subscribe("alerts.scored", this::notifyConsole);
    }

    void notifyConsole(LogEvent ev) {
        System.out.println(
                "[ALERT] " + ev.getId() + " risk=" + ev.getFeatures().get("final_risk") + " msg=" + ev.getMessage());
    }
}