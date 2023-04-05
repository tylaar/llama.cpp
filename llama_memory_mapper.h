//
// Created by Yifeng Yu on 2023/4/5.
//

#ifndef LLAMA_CPP_LLAMA_MEMORY_MAPPER_H
#define LLAMA_CPP_LLAMA_MEMORY_MAPPER_H


#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

class llama_memory_mapper {
public:
    static void *mmap_file(const char *fname, uint64_t *mm_length) {
#if defined(_WIN32) && !defined(_POSIX_MAPPED_FILES)
        HANDLE hFile = CreateFileA(fname,
                               GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               NULL,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED,
                               NULL);
    if (hFile == INVALID_HANDLE_VALUE) return 0;
    LARGE_INTEGER fileSize;
    fileSize.QuadPart = -1;
    GetFileSizeEx(hFile, &fileSize);
    int64_t length = fileSize.QuadPart;
    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(hFile);
    if (!hMapping) return 0;
    void *addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);
    if (!addr) return 0;
#else
        int fd = open(fname, O_RDONLY);
        if (fd == -1) return 0;
        int64_t length = lseek(fd, 0, SEEK_END);
        void *addr = mmap(NULL, length, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (addr == MAP_FAILED) return 0;
#endif
        *mm_length = length;
        return addr;
    }

    static void munmap_file(void *addr, size_t length) {
#if defined(_WIN32) && !defined(_POSIX_MAPPED_FILES)
        UnmapViewOfFile(addr);
#else
        munmap(addr, length);
#endif
    }


};


#endif //LLAMA_CPP_LLAMA_MEMORY_MAPPER_H
