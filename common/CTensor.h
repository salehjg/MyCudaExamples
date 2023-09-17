//
// Created by saleh on 9/17/23.
//

#ifndef FARADARSCUDABASICS_CTENSOR_H
#define FARADARSCUDABASICS_CTENSOR_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

#define CHECK(E) if(E!=cudaError_t::cudaSuccess) std::cerr<<"CUDA API FAILED, File: "<<__FILE__<<", Line: "<< __LINE__ << ", Error: "<< cudaGetErrorString(E) << std::endl;

enum class FillTypes {
    kRandom,
    kIncr,
    kDecr,
    kCustom1,
};

template<typename T>
class CTensor {
protected:
    const std::vector<size_t> m_vShape;
    const size_t m_ulSize, m_ulSizeBytes;
    T *m_ptrDataDevice;
    T *m_ptrDataHost;

    static size_t _GetSize(const std::vector<size_t> &shape) {
        size_t size = 1;
        std::for_each(shape.begin(), shape.end(), [&size](size_t val) { size *= val; });
        return size;
    }

    static size_t _GetSizeBytes(const std::vector<size_t> &shape) {
        return _GetSize(shape) * sizeof(T);
    }

    static T RandomValue(T minVal, T maxVal) {
        std::mt19937 rng;
        std::uniform_real_distribution<float> u(minVal, maxVal);
        return u(rng);
    }

public:
    CTensor(const std::vector<size_t> &shape) : m_vShape(shape), m_ulSize(_GetSize(shape)),
                                                m_ulSizeBytes(_GetSizeBytes(shape)) {
        m_ptrDataHost = new T[m_ulSize];
        CHECK(cudaMalloc((void**)&m_ptrDataDevice, m_ulSizeBytes));
    }

    ~CTensor() {
        delete[] m_ptrDataHost;
        CHECK(cudaFree(m_ptrDataDevice));
    }

    T &operator[](size_t rowMajorIndex) {
        return m_ptrDataHost[rowMajorIndex];
    }

    void H2D() {
        // dest ptr, src ptr, size bytes, enum
        CHECK(cudaMemcpy(m_ptrDataDevice, m_ptrDataHost, m_ulSizeBytes, cudaMemcpyHostToDevice));
    }

    void D2H() {
        // dest ptr, src ptr, size bytes, enum
        CHECK(cudaMemcpy(m_ptrDataHost, m_ptrDataDevice, m_ulSizeBytes, cudaMemcpyDeviceToHost));
    }

    void Fill(const FillTypes &type) {
        switch (type) {
            case FillTypes::kRandom: {
                for (size_t i = 0; i < m_ulSize; i++) { m_ptrDataHost[i] = RandomValue(0.0f, 100.0f); }
                break;
            }
            case FillTypes::kIncr: {
                for (size_t i = 0; i < m_ulSize; i++) { m_ptrDataHost[i] = (T) i; }
                break;
            }
            case FillTypes::kDecr: {
                for (size_t i = 0; i < m_ulSize; i++) { m_ptrDataHost[i] = (T) (m_ulSize - i - 1); }
                break;
            }
            case FillTypes::kCustom1: {
                for (size_t i = 0; i < m_ulSize; i++) { m_ptrDataHost[i] = (T) (i % 10); }
                break;
            }
            default: {
                std::cerr << "Unknown FillTypes." << std::endl;
                return;
            }
        }
        H2D();
    }

    size_t GetSize() {
        return m_ulSize;
    }

    size_t GetSizeBytes() {
        return m_ulSizeBytes;
    }

    T* GetPtrHost() {return m_ptrDataHost;}
    T* GetPtrDevice() {return m_ptrDataDevice;}
};


#endif //FARADARSCUDABASICS_CTENSOR_H
