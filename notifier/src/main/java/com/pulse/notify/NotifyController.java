package com.pulse.notify;

import com.pulse.common.dto.LogEvent;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/v1/notify")
@RequiredArgsConstructor
public class NotifyController {

    @PostMapping
    public String notifyNow(@RequestBody LogEvent ev) {
        System.out.println("[ALERT] id=" + ev.getId()
                + " risk=" + ev.getFeatures().get("final_risk")
                + " msg=" + ev.getMessage());
        return "sent";
    }
}
