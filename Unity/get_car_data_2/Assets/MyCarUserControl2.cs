//研究用MyCarUserControl.cs
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
    public class MyCarUserControl2 : MonoBehaviour
    {
        private CarController m_Car; // the car controller we want to use
                                     // private GetParameter m_UI = GetComponent<GetParameter>();
        Text pText;
        GameObject car;
        GameObject spawner;
        Vector2 speedVector, pos, oldPos;
        Vector2[] wheelVector = new Vector2[2];
        Quaternion initAngle = Quaternion.identity;
        public GameObject dot;
        public WheelCollider[] wheel = new WheelCollider[4];
        public GameObject[] wheelHub = new GameObject[4];
        public Transform[] point = new Transform[3];
        public float endtime = 300.0f;
        public string fileName = "default";
        int count = 0, carStatementNum = 0, dataNum = 0, teacherDataNum = 0;
        int[] isSlip = new int[4];
        float[] time = new float[5];
        float[] slipAngle = new float[2];
        float[] slipRatio = new float[4];
        float[] pastSlipRatio = new float[4];
        float[] pastSlipAngle = new float[4];
        float slipLimit = 0.0f;
        float acceleration = 0.0f;
        float speed = 0.0f, limitedSpeed = 0.0f;
        float Prevspeed = 0.0f;
        float wheelSpeedMS = 0.0f;
        float carSpeedMS = 0.0f;
        float h = 0.0f;
        float v = 0.0f;
        float deltaH = 0.0f;
        float deltaSpeed = 0.0f;
        float x = 0, z = 0;
        string carStatement, dataString;
        string teacherDataPath = "C:\\Users\\odyssey8942\\Documents\\new_get_car_data\\CarData\\";

        void Start()
        {
            pText = GameObject.Find("ParameterText").GetComponent<Text>();
            car = GameObject.Find("Car");
            spawner = GameObject.Find("Spawner");
        }
        private void Awake()
        {
            // get the car controller
            m_Car = GetComponent<CarController>();
        }
        public void FixedUpdate()
        {
            float h = CrossPlatformInputManager.GetAxis("Horizontal");
            float v = -1*CrossPlatformInputManager.GetAxis("L_R_Trigger");
            float handbrake = CrossPlatformInputManager.GetAxis("Submit");
            m_Car.Move(h, v, v, handbrake);
            // pass the input to the car!
            for (int i = 0; i < 5; i++)
            {
                time[i] += Time.deltaTime;
            }
            pText.text = "time(sec) : " + time[0]
                        + "\nアクセル開度 : " + (Mathf.Clamp(v, -1, 1) * 100.0f)
                        + "\n舵角 : " + m_Car.CurrentSteerAngle
                        + "\n速度(km/s) : " + speed
                        + "\n加速度 : " + acceleration
                        + "\nトルク : " + m_Car.CurrentTorque
                        + "\n前輪スリップ角 : " + slipAngle[0]
                        + "\n後輪スリップ角 : " + slipAngle[1]
                        + "\n左前輪スリップ率 : " + slipRatio[0]
                        + "\n右前輪スリップ率 : " + slipRatio[1]
                        + "\n左後輪スリップ率 : " + slipRatio[2]
                        + "\n右後輪スリップ率 : " + slipRatio[3]
                        + "\n車両状態 : " + carStatement
                        ;
            
            if(time[0] >= endtime)
                SceneManager.LoadScene("taskcompleted");
            //軌跡表示
            if (time[1] > 0.1f)
            {
                time[1] = 0.0f;
                Instantiate(dot, this.transform.position, Quaternion.identity);
            }

            //パラメータ取得
            if (time[2] > 0.1f)
            {
                time[2] = 0.0f;
                //速度,加速度,0.1秒前の速度を取得
                speed = m_Car.CurrentSpeed * 1.6f;
                acceleration = (speed - Prevspeed) / 0.1f;
                Prevspeed = speed;
                //現在,0.1秒前の自動車の座標を取得
                pos = new Vector2(car.transform.position.x, car.transform.position.z);
                speedVector = new Vector2((pos.x - oldPos.x) / 0.1f, (pos.y - oldPos.y) / 0.1f);
                oldPos = new Vector2(car.transform.position.x, car.transform.position.z);
            }
            //タイヤの角度を取得
            wheelVector[0] = new Vector2(point[0].position.x - point[1].position.x, point[0].position.z - point[1].position.z);
            wheelVector[1] = new Vector2(point[1].position.x - point[2].position.x, point[1].position.z - point[2].position.z);
            wheelVector[0] = Quaternion.Euler(0f, h * 25f, 0f) * wheelVector[0];

            for (int i = 0; i < 2; i++)
            {
                //スリップ角を取得
                slipAngle[i] = Mathf.Acos(Vector2.Dot(wheelVector[i], speedVector) / (wheelVector[i].magnitude * speedVector.magnitude)) * 180 / Mathf.PI;

                if (float.IsNaN(slipAngle[i]))
                    slipAngle[i] = pastSlipAngle[i];
                else
                    pastSlipAngle[i] = slipAngle[i];

            }

            for (int i = 0; i < 4; i++)
            {
                //速度をkm/sに変換
                carSpeedMS = speed * 1000 / 3600;
                //車輪速度を取得
                wheelSpeedMS = 2 * Mathf.PI * wheel[i].radius * (wheel[i].rpm / 60);
                //スリップ率を取得
                if (v > 0)
                    slipRatio[i] = (wheelSpeedMS - carSpeedMS) / wheelSpeedMS;
                else
                    slipRatio[i] = (carSpeedMS - wheelSpeedMS) / carSpeedMS;

                if (float.IsInfinity(slipRatio[i]))
                    slipRatio[i] = pastSlipRatio[i];
                else
                    pastSlipRatio[i] = slipRatio[i];
            }

            //スリップ角、スリップ率、スタビリティーファクター判定
            if (time[3] > 1.0)
            {
                DecisionSlipAngle();
            }
            DecisionSlipRatio();

            //データファイル出力
            if (time[4] > 0.1f)
            {
                time[4] = 0.0f;
                WriteFile(Mathf.Clamp(v, -1, 1));
            }
        }
        //ファイル書き込み
        void WriteFile(float v)
        {
            dataString = dataNum++ + ","
                       + v + ","
                       + m_Car.CurrentSteerAngle + ","
                       + speed + ","
                       + acceleration + ","
                       + m_Car.CurrentTorque + ","
                       + slipAngle[0] + ","
                       + slipAngle[1] + ","
                       + slipRatio[0] + ","
                       + slipRatio[1] + ","
                       + slipRatio[2] + ","
                       + slipRatio[3] + ","
                       + carStatementNum + ","
                       + isSlip[0] + ","
                       + isSlip[1] + ","
                       + isSlip[2] + ","
                       + isSlip[3] + "\n"
                       ;
            File.AppendAllText(teacherDataPath + fileName +".csv", dataString);
        }
        //車両状態判定
        void DecisionSlipAngle()
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
                    carStatementNum = 2;
                }
                else if (slipAngle[1] > slipAngle[0])
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
            else if (slipAngle[0] > 10 || slipAngle[1] > 10)
            {
                if (slipAngle[0] > slipAngle[1])
                {
                    carStatement = "強アンダーステア";
                    carStatementNum = 3;
                }
                else if (slipAngle[1] > slipAngle[0])
                {
                    carStatement = "強オーバーステア";
                    carStatementNum = -3;
                }
                else
                {
                    carStatement = "ニュートラルステア";
                    carStatementNum = 0;
                }
            }
            else { }
        }
        void DecisionSlipRatio()
        {
            for (int i = 0; i < 4; i++)
            {
                if (slipRatio[i] > 0.20)
                    isSlip[i] = 1;
                else
                    isSlip[i] = 0;
            }
        }
    }
}


