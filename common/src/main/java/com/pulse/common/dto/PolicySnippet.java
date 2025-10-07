package com.pulse.common.dto;

import java.util.Objects;

/**
 * Simple POJO with no Lombok assumptions so Jackson always works.
 */
public class PolicySnippet {
    private String text;
    private double similarity;
    private String sectionRef;

    public PolicySnippet() {
    } // required by Jackson

    public PolicySnippet(String text, double similarity, String sectionRef) {
        this.text = text;
        this.similarity = similarity;
        this.sectionRef = sectionRef;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public double getSimilarity() {
        return similarity;
    }

    public void setSimilarity(double similarity) {
        this.similarity = similarity;
    }

    public String getSectionRef() {
        return sectionRef;
    }

    public void setSectionRef(String sectionRef) {
        this.sectionRef = sectionRef;
    }

    @Override
    public String toString() {
        return "PolicySnippet{text='%s', similarity=%s, sectionRef='%s'}"
                .formatted(text, similarity, sectionRef);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof PolicySnippet that))
            return false;
        return Double.compare(that.similarity, similarity) == 0 &&
                Objects.equals(text, that.text) &&
                Objects.equals(sectionRef, that.sectionRef);
    }

    @Override
    public int hashCode() {
        return Objects.hash(text, similarity, sectionRef);
    }
}
