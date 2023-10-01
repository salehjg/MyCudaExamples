//
// Created by saleh on 10/1/23.
//

#pragma once

#include <random>

template <typename T>
class CRandFiller {
public:
    CRandFiller(T min, T max) {
        m_min = min;
        m_max = max;
        m_uni = std::uniform_real_distribution<T>(m_min, m_max);
    }

    T GetRand() {
        return m_uni(m_eng);
    }

protected:
    std::mt19937 m_eng;
    std::uniform_real_distribution<T> m_uni;
    T m_min, m_max;
};