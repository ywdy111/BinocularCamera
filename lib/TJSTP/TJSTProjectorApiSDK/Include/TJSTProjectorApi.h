#pragma once
/*
* 文件名：[TJSTProjectorApi.h]
* 作者：〈杭州腾聚科技-JXB〉
* 描述：〈光机开发包〉
* 修改人：〈JXB〉
* 修改时间：2020 - 08 - 19
* 修改内容：〈新建〉

* 修改人：〈JXB〉
* 修改时间：2020 - 08 - 25
* 修改内容：〈修改〉
	1.增加固定IP网络设备的打开函数：TJSTNetPrjOpen

*/
#ifdef _LINUX_OS_
#define TJPRJ_API __attribute__((visibility("default")))
#else
#ifdef TJSTPROJECTORAPI_EXPORTS
#define TJPRJ_API extern "C" __declspec(dllexport)
#else
#define TJPRJ_API extern "C" __declspec(dllimport)
#endif
#endif

typedef enum
{
	PRJ_TYPE_USBPOD = 0, //USBPOD设备
	PRJ_TYPE_USBHID,	 //USBHID设备
	PRJ_TYPE_NET,		 //网络设备
}TJSTPrjType_t;

typedef struct
{
	TJSTPrjType_t prjType;//设备类型
	int prjIndex;		  //序号
	union info
	{
		int prjID; //USBPOD设备或者USBHID设备有效：<=0表示没有指定ID；>0 指定ID值 (同时连接多套设备时用于彼此区分)
		unsigned char prjIP[4];//网络设备：IP 地址（IPV4).
	}prjInfo;
	char prjVer[64];//设备版本字符串
}TJSTPrjInfo_t;

typedef void* TJSTPRJ; //设备描述符

/*
函数：TJSTEnumDevices
功能：枚举所有设备
返回：设备数量
*/
TJPRJ_API int TJSTEnumDevices();

/*
函数：TJSTPrjOpenIndex
功能：打开指定设备
参数：nIndex：设备序号（从0开始）
返回：
	成功：设备描述符
	失败：0
*/
TJPRJ_API TJSTPRJ TJSTPrjOpenIndex(int nIndex);


/*
函数：TJSTGetPrjInfo
功能：取得指定设备信息
参数：nIndex：第几个设备。0为第一个设备
返回：
	成功：设备信息结构体指针
	失败：0
*/
TJPRJ_API const TJSTPrjInfo_t* TJSTGetPrjInfo(int nIndex);

/*
函数：TJSTPrjOpen
功能：打开指定设备
参数：prjInfo：设备信息结构体指针，由TJSTGetPrjInfo获得
返回：
	成功：设备描述符
	失败：0
*/
TJPRJ_API TJSTPRJ TJSTPrjOpen(const TJSTPrjInfo_t* prjInfo);

/*
函数：TJSTNetPrjOpen
功能：打开固定IP网络设备。注意：如果设备不存在，此可能会卡死30秒
参数：prjIP：设备IPV4地址字符串.(比如"192.168.100.100")
返回：
	成功：设备描述符
	失败：0
*/
TJPRJ_API TJSTPRJ TJSTNetPrjOpen(const char* prjIP);

/*
函数：TJSTNetPrjOpenEx
功能：打开固定IP网络设备。注意：如果设备不存在，此可能会卡死30秒
参数：
	prjIP：设备IPV4地址字符串.(比如"192.168.100.100")
	nPort: 设备端口，默认1234
返回：
成功：设备描述符
失败：0
*/
TJPRJ_API TJSTPRJ TJSTNetPrjOpenEx(const char* prjIP,int nPort);
/*
函数：TJSTPrjClose
功能：关闭指定设备
参数：prjInfo：设备信息结构体指针，由TJSTGetPrjInfo获得
*/
TJPRJ_API void TJSTPrjClose(TJSTPRJ prj);

/*
函数：TJSTPrjWrite
功能：向设备写入命令
参数：prj：设备描述符
	  pWBuff：命令字符指针
	  nWLen：命令长度（字节数）
返回：
	成功：写入的字节数
	失败：0
*/
TJPRJ_API int TJSTPrjWrite(TJSTPRJ prj, const char* pWBuff, int nWLen);

/*
函数：TJSTPrjCmdNotBack
功能：明确指示上一个命令没有数据返回
参数：prj：设备描述符
*/
TJPRJ_API void TJSTPrjCmdNotBack(TJSTPRJ prj);
/*
函数：TJSTPrjRead
功能：从设备读取返回信息
参数：prj：设备描述符
	pRBuff：读取字符指针，由用户申请和释放内存
	nRLen：读取长度（字节数），不能大于pRBuff内存大小
返回：
	成功：读取的字节数
	失败：0
注意：USBHID设备,在没有数据返回时调用TJSTPrjRead 函数会导致程序卡死(TJSTPrjRead 无法返回)
*/
TJPRJ_API int TJSTPrjRead(TJSTPRJ prj, char* pRBuff, int nRLen);

/*
函数：TJSTPrjLedOn
功能：打开设备投影灯
参数：prj：设备描述符
返回：
	成功：true
	失败：false
*/
TJPRJ_API bool TJSTPrjLedOn(TJSTPRJ prj);

/*
函数：TJSTPrjLedOff
功能：关闭设备投影灯。在不工作时关闭投影灯有助于延长设备使用寿命。
参数：prj：设备描述符
返回：
	成功：true
	失败：false
*/
TJPRJ_API bool TJSTPrjLedOff(TJSTPRJ prj);

/*
函数：TJSTPrjSetMode
功能：设置设备投影内容模式
参数：prj：设备描述符
	  nMode：内容模式. 0 黑屏；1 白屏；2 十字 ；3 棋盘
返回：
	成功：true
	失败：false
*/
TJPRJ_API bool TJSTPrjSetMode(TJSTPRJ prj, unsigned char nMode);

/*
函数：TJSTPrjSetColor
功能：设置设备投影。只有多光谱结构光投影机支持
参数：prj：设备描述符
	  nColor：颜色. 0 红色；1 绿色；2 蓝色 ；3 白色
返回：
	成功：true
	失败：false
*/
TJPRJ_API bool TJSTPrjSetColor(TJSTPRJ prj, unsigned char nColor);


/*
函数：TJSTPrjSetLight
功能：设置投影亮度。亮度越高发热越严重。在不工作时关闭投影灯有助于延长设备使用寿命。
参数：prj：设备描述符
	  nLight：亮度值，范围（10-200）。设置高亮度（大于100）注意散热
返回：
	成功：true
	失败：false
*/
TJPRJ_API bool TJSTPrjSetLight(TJSTPRJ prj, unsigned char nLight);


/*
函数：TJSTPrjTriggerOnce
功能：给设备发送触发(投射条纹)命令
参数：prj：设备描述符
	  nGray：条纹末尾投射灰度值（0-255）。比如下载了12幅条纹，并且配置了触发13幅条纹(MA 0 12 0 0)，那么第13幅投射的亮度由nGray确定，0为黑色，255为白色
	  (部分机型nGray支持0-254）
返回：
	成功：true
	失败：false
*/
TJPRJ_API bool TJSTPrjTriggerOnce(TJSTPRJ prj, unsigned char nGray);
