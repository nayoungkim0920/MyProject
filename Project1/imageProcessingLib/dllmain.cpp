// dllmain.cpp : DLL 응용 프로그램의 진입점을 정의합니다.
#include "pch.h"

BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved
)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        // DLL이 프로세스에 로드될 때 실행
        break;
    case DLL_THREAD_ATTACH:
        // DLL이 스레드에 로드될 때 실행
        break;
    case DLL_THREAD_DETACH:
        // 스레드에서 DLL이 언로드될 때 실행
        break;
    case DLL_PROCESS_DETACH:
        // 프로세스에서 DLL이 언로드될 때 실행
        break;
    }
    return TRUE;
}
