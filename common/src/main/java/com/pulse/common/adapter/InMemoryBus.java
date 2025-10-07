package com.pulse.common.adapter;
import com.pulse.common.port.MessageBus;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;

public class InMemoryBus<T> implements MessageBus<T> {
private final Map<String, List<Consumer<T>>> subscribers = new ConcurrentHashMap<>();

@Override public void publish(String topic, T message) {
subscribers.getOrDefault(topic, List.of()).forEach(c -> c.accept(message));
}
@Override public void subscribe(String topic, Consumer<T> consumer) {
subscribers.computeIfAbsent(topic, k -> new CopyOnWriteArrayList<>()).add(consumer);
}
}