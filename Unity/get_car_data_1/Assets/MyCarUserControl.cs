using System;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEditor;
using System.Collections;
using System.Collections.Generic;
using UnityStandardAssets.CrossPlatformInput;
using UnityEngine.UI;
using System.IO;
namespace UnityStandardAssets.Vehicles.Car
{
    [RequireComponent(typeof(CarController))]
    public class MyCarUserControl : MonoBehaviour
    {
        private CarController m_Car; // the car controller we want to use
                                     // private GetParameter m_UI = GetComponent<GetParameter>();
        Text pText;
        GameObject car;
        GameObject pillar;
        GameObject spawner;
        Vector3 startpos,newPillarPos,farPos,comparePos;
        Vector2 speedVector,pos,oldPos;
        Vector2[] wheelVector = new Vector2[2];
        Quaternion initAngle = Quaternion.identity; 
        public GameObject dot;
        public WheelCollider[] wheel = new WheelCollider[4];
        public GameObject[] wheelHub = new GameObject[4];
        public Transform[] point = new Transform[3];
        Boolean setSpeed = false, setPos = false, ready = false, setTime2 = false, resetScene = false;
        int count = 0, carStatementNum = 0, dataNum = 0, teacherDataNum = 0;
        private int topSpeed = 100, maxTorque = 6000;
        private float maxSlip = 0.8f;
        int[] isSlip = new int[4];
        float[] time = new float[10];
        float[] slipAngle = new float[2];
        float[] slipRatio = new float[4];
        float r0 = 0.0f;
        float r = 0.0f;
        float farDis = 0.0f;
        float compareDis = 0.0f;
        float targetSpeed = 0.0f;
        float targetTorque = 0.0f;
        float slipLimit = 0.0f;
        float acceleration = 0.0f;
        float speed = 0.0f, limitedSpeed = 0.0f;
        float Prevspeed = 0.0f;
        float wheelSpeedMS = 0.0f;
        float carSpeedMS = 0.0f;
        float stabilitiyFactor = 0.0f;
        float h = 0.0f;
        float hh = 0.0f;
        float hhh = 0.0f;
        float v = 0.0f;
        float deltaH = 0.0f;
        float deltaSpeed = 0.0f;
        float x = 0, z = 0;
        string carStatement, dataString, targetStr, hhStr, teacherDataNumStr, torqueStr, slipLimitStr;
        string teacherDataPath   = "C:\\Users\\odyssey8942\\Documents\\get_car_data\\TeacherData\\";
        string parameterDataPath = "C:\\Users\\odyssey8942\\Documents\\get_car_data\\ParameterData\\Parameter.txt";
        string formatDataPath    = "C:\\Users\\odyssey8942\\Documents\\get_car_data\\ParameterData\\Format.txt";
        FileInfo fi = new FileInfo("C:\\Users\\odyssey8942\\Documents\\get_car_data\\ParameterData\\Parameter.txt");

        void Start()
        {
            pText = GameObject.Find("ParameterText").GetComponent<Text>();
            car = GameObject.Find("Car");
            pillar = GameObject.Find("Pillar");
            spawner = GameObject.Find("Spawner");

            ReadFile();
            teacherDataNum = int.Parse(teacherDataNumStr);
            hh = float.Parse(hhStr);
            targetSpeed = float.Parse(targetStr);
            targetTorque = float.Parse(torqueStr);
            slipLimit = float.Parse(slipLimitStr);
            if (hh < 1.0f)
            {
                hh += 0.1f;
            }
            else if (hh >= 1.0f)
            {
                hh = -1.0f;
                targetSpeed += 10.0f;
            }

            if(targetSpeed > topSpeed)
            {
                hh = -1.0f;
                targetSpeed = 10.0f;
                targetTorque += 2000;
            }
            if(targetTorque > maxTorque)
            {
                hh = -1.0f;
                targetSpeed = 10.0f;
                targetTorque = 2000;
                slipLimit += 0.3f;
            }
            File.Delete(parameterDataPath);
            File.AppendAllText(parameterDataPath, (teacherDataNum+1).ToString() +"\r\n" + hh.ToString() + "\r\n" + targetSpeed.ToString() + "\r\n" + targetTorque.ToString() + "\r\n" + slipLimit.ToString());
            hh = float.Parse(hhStr);
            targetSpeed = float.Parse(targetStr);
            targetTorque = float.Parse(torqueStr);
            slipLimit = float.Parse(slipLimitStr);
            m_Car.m_SlipLimit = slipLimit;
            m_Car.m_FullTorqueOverAllWheels = targetTorque;
            if (hh == 0.0f)
            {
                SceneManager.LoadScene("getcardata");
            }
        }
        private void Awake()
        {
            // get the car controller
            m_Car = GetComponent<CarController>();
        }


        public void FixedUpdate()
        {

            // pass the input to the car!
            for (int i = 0; i < 10; i++)
            {
                    time[i] += Time.deltaTime;
            }
            pText.text = "time(sec) : " + time[9]
                        + "\nアクセル開度 : " + (Mathf.Clamp(v, 0, 1) * 100.0f)
                        + "\n舵角 : " + m_Car.CurrentSteerAngle
                        + "\n速度(km/s) : " + speed
                        + "\n加速度 : " + acceleration
                        + "\nトルク : " + m_Car.CurrentTorque
                        + "\n前輪角 : " + (car.transform.eulerAngles.y + h * 25)
                        + "\n後輪角 : " + car.transform.eulerAngles.y
                        + "\n前輪スリップ角 : " + slipAngle[0]
                        + "\n後輪スリップ角 : " + slipAngle[1]
                        + "\nスタビリティーファクター : " + stabilitiyFactor
                        + "\n左前輪スリップ率 : " + slipRatio[0]
                        + "\n右前輪スリップ率 : " + slipRatio[1]
                        + "\n左後輪スリップ率 : " + slipRatio[2]
                        + "\n右後輪スリップ率 : " + slipRatio[3]
                        + "\n車両状態 : " + carStatement
                        + "\n路面摩擦 : " + m_Car.m_SlipLimit
                        ;
#if !MOBILE_INPUT
            float handbrake = CrossPlatformInputManager.GetAxis("Jump");
            m_Car.Move(h, v, v, handbrake);
            
            //テスト開始までの判定処理
            if (setSpeed == true)
            {
                h = hh;
                if (setPos == false)
                {
                    if (time[2] > 3.0)
                    {
                        setPos = true;
                        startpos = new Vector3(car.transform.position.x, car.transform.position.y, car.transform.position.z);
                    }
                }
                if (compareDis > farDis)
                {
                    farDis = compareDis;
                    farPos = car.transform.position;
                }
                if (time[2] > 3.0)
                {
                    comparePos = new Vector3(startpos.x - car.transform.position.x, startpos.y - car.transform.position.y, startpos.z - car.transform.position.z);
                    compareDis = comparePos.sqrMagnitude;
                }          
                if (Mathf.Abs(car.transform.position.z - startpos.z) < 2.0f && Mathf.Abs(car.transform.position.x - startpos.x) < 2.0f && farDis > 10)
                {
                    if (ready == false)
                    {
                        r0 = (Mathf.Sqrt(farDis))/2;
                        x = (farPos.x + startpos.x)/2;
                        z = (farPos.z + startpos.z)/2;
                        newPillarPos = new Vector3(x ,0, z );
                        pillar.transform.position = newPillarPos;
                        v = targetSpeed / m_Car.MaxSpeed;
                        ready = true;
                        time[9] = 0.0f;
                    }
                }
            }
            else
            {
                startpos = car.transform.position;
            }
            
            //アクセル操作
            if (time[1] > 0.01f)
            {
                if (ready == true)
                    v += 0.0005f;
                else if (ready == false)
                {
                    if ( speed < targetSpeed)
                        v += 0.1f;
                    else
                    {
                        if (setSpeed == false)
                        {
                            time[9] = 0.0f;
                            setSpeed = true;
                        }
                        if (setTime2 == false)
                        {
                            time[2] = 0;
                            setTime2 = true;
                        }
                        v = targetSpeed / m_Car.MaxSpeed;
                    }
                }
            }
            //軌跡表示
            if(time[3] > 0.1f)
            {
                time[3] = 0.0f;
                Instantiate(dot, this.transform.position, Quaternion.identity);
            }

            //パラメータ取得
            if (time[0] > 0.1f)
            {
                time[0] = 0.0f;

                speed = m_Car.CurrentSpeed * 1.6f;
                acceleration = (speed - Prevspeed) / 0.1f;
                Prevspeed = speed;

                pos = new Vector2(car.transform.position.x, car.transform.position.z);
                speedVector = new Vector2((pos.x - oldPos.x) / 0.1f, (pos.y - oldPos.y) / 0.1f);
                oldPos = new Vector2(car.transform.position.x, car.transform.position.z); 
            }
            wheelVector[0] = new Vector2(point[0].position.x - point[1].position.x, point[0].position.z - point[1].position.z);
            wheelVector[1] = new Vector2(point[1].position.x - point[2].position.x, point[1].position.z - point[2].position.z);
            wheelVector[0] = Quaternion.Euler(0f, h * 25f, 0f) * wheelVector[0];
            for (int i = 0; i < 2; i++)
            {
                slipAngle[i] = Mathf.Acos(Vector2.Dot(wheelVector[i], speedVector) / (wheelVector[i].magnitude * speedVector.magnitude)) * 180 / Mathf.PI;
            }

            for (int i = 0; i < 4; i++)
            {
                carSpeedMS = speed * 1000 / 3600;
                wheelSpeedMS = 2 * Mathf.PI * wheel[i].radius * (wheel[i].rpm / 60);
                if (v > 0)
                    slipRatio[i] = (wheelSpeedMS - carSpeedMS) / wheelSpeedMS;
                else
                    slipRatio[i] = (carSpeedMS - wheelSpeedMS) / carSpeedMS;
            }
            if(ready == true)
                r = Vector3.Magnitude(car.transform.position - pillar.transform.position);
            
            //スリップ角、スリップ率、スタビリティーファクター判定
            if (time[8] > 1.0)
            {
                DecisionSlipAngle();
            }
            DecisionSlipRatio();

            //データファイル出力
            if (time[4] > 0.1f)
            {
                time[4] = 0.0f;
                WriteFile();
            }

            if (resetScene == true)
            {
                WriteFile();
                SceneManager.LoadScene("getcardata");
            }
            if (time[9] > 120 && ready == true) { SceneManager.LoadScene("getcardata"); }
            else if(time[9] > 120 && setSpeed == false && ready == false) { SceneManager.LoadScene("getcardata"); }
            else if(time[9] > 180 && setSpeed == true && ready == false) { SceneManager.LoadScene("getcardata"); }
            else { }

            if(slipLimit > maxSlip )
                SceneManager.LoadScene("taskcompleted");
#else
            m_Car.Move(h, v, v, 0f);
#endif
        }

        void ReadFile()
        {
            // FileReadTest.txtファイルを読み込む
            try
            {
                // 一行毎読み込み
                using (StreamReader sr = new StreamReader(fi.OpenRead(), Encoding.UTF8))
                {
                    teacherDataNumStr = sr.ReadLine();
                    hhStr = sr.ReadLine();
                    targetStr = sr.ReadLine();
                    torqueStr = sr.ReadLine();
                    slipLimitStr = sr.ReadLine();
                }
            }
            catch (Exception e)
            {

            }
        }
        void WriteFile()
        {
            dataString = dataNum++ + " "
                       + (Mathf.Clamp(v, 0, 1) * 100.0f) + " "
                       + m_Car.CurrentSteerAngle + " "
                       + speed + " "
                       + acceleration + " "
                       + m_Car.CurrentTorque + " "
                       + slipAngle[0] + " "
                       + slipAngle[1] + " "
                       + stabilitiyFactor + " "
                       + slipLimit + " "
                       + slipRatio[0] + " "
                       + slipRatio[1] + " "
                       + slipRatio[2] + " "
                       + slipRatio[3] + " "
                       + carStatementNum + " "
                       + isSlip[0] + " "
                       + isSlip[1] + " "
                       + isSlip[2] + " "
                       + isSlip[3] + "\n"
                       ;
            File.AppendAllText(teacherDataPath + teacherDataNum + ".csv", dataString);
        }
        void DecisionSlipAngle()
        {
            if(ready == true)
                {
                stabilitiyFactor = ((r / r0) - 1) / speed * speed;
                if (slipAngle[0] <= 4 && slipAngle[1] <= 4)
                {
                    if (stabilitiyFactor > 0)
                    {
                        carStatement = "弱アンダーステア";
                        carStatementNum = 1;
                    }
                    else if (stabilitiyFactor < 0)
                    {
                        carStatement = "弱オーバーステア";
                        carStatementNum = -1;
                    }
                    else
                    {
                        carStatement = "ニュートラルステア";
                        carStatementNum = 0;
                    }
                }
                else if ((slipAngle[0] > 4 || slipAngle[1] > 4) && (slipAngle[0] <= 10 && slipAngle[1] <= 10))
                {
                    if (stabilitiyFactor > 0)
                    {
                        carStatement = "中アンダーステア";
                        carStatementNum = 2;
                    }
                    else if (stabilitiyFactor < 0)
                    {
                        carStatement = "中オーバーステア";
                        carStatementNum = -2;
                    }
                    else
                    {
                        carStatement = "ニュートラルステア";
                        carStatementNum = 0;
                    }
                }
                else if ((slipAngle[0] > 10 || slipAngle[1] > 10) && speed > 30.0f)
                {
                    if (stabilitiyFactor > 0)
                    {
                        carStatement = "強アンダーステア";
                        carStatementNum = 3;
                        resetScene = true;
                    }
                    else if (stabilitiyFactor < 0)
                    {
                        limitedSpeed = Mathf.Sqrt(-1 / stabilitiyFactor);
                        if (speed > limitedSpeed)
                            carStatement = "過オーバーステア";
                        else
                            carStatement = "強オーバーステア";
                        carStatementNum = -3;
                        resetScene = true;
                    }
                    else
                    {
                        carStatement = "ニュートラルステア";
                        carStatementNum = 0;
                    }
                    /*
                    if (stabilitiyFactor < 0)
                    {
                        limitedSpeed = Mathf.Sqrt(-1 / stabilitiyFactor);
                        if (speed > limitedSpeed)
                        carStatement = "円旋回不可能";
                    }
                    */
                }
                else { }
            }
            else
                {
                if (slipAngle[0] <= 4 && slipAngle[1] <= 4)
                {
                    if (slipAngle[0] > slipAngle[1])
                    {
                        carStatement = "弱アンダーステア";
                        carStatementNum = 1;
                    }
                    else if (slipAngle[1] > slipAngle[0])
                    {
                        carStatement = "弱オーバーステア";
                        carStatementNum = -1;
                    }
                    else
                    {
                        carStatement = "ニュートラルステア";
                        carStatementNum = 0;
                    }
                }
                else if ((slipAngle[0] > 4 || slipAngle[1] > 4) && (slipAngle[0] <= 10 && slipAngle[1] <= 10))
                {
                    if (slipAngle[0] > slipAngle[1])
                    {
                        carStatement = "中アンダーステア";
                        carStatementNum = 1;
                    }
                    else if (slipAngle[1] > slipAngle[0])
                    {
                        carStatement = "中オーバーステア";
                        carStatementNum = -1;
                    }
                    else
                    {
                        carStatement = "ニュートラルステア";
                        carStatementNum = 0;
                    }
                }
                else if ((slipAngle[0] > 10 || slipAngle[1] > 10) && speed > 30.0f)
                {
                    if (slipAngle[0] > slipAngle[1])
                    {
                        carStatement = "強アンダーステア";
                        carStatementNum = 3;
                        resetScene = true;
                    }
                    else if (slipAngle[1] > slipAngle[0])
                    {
                        carStatement = "強オーバーステア";
                        carStatementNum = -3;
                        resetScene = true;
                    }
                    else
                    {
                        carStatement = "ニュートラルステア";
                        carStatementNum = 0;
                    }
                }
                else { }
            }
        }
        void DecisionSlipRatio()
        {
            for(int i = 0; i < 4; i++)
            {
                if (slipRatio[i] > 0.20)
                    isSlip[i] = 1;
                else
                    isSlip[i] = 0;
            }
        }
    }
}


