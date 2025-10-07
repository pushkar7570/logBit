package com.pulse.common.port;
import java.util.function.Consumer;

public interface MessageBus<T> {
void publish(String topic, T message);
void subscribe(String topic, Consumer<T> consumer);
}