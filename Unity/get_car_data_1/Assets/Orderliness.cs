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

public class Orderliness : MonoBehaviour {
    int i = 0, j = 0,k = 0;
    int start = 1;
    int end = 5400;
    int count = 540;
    int[] dataNum = new int[540];
    string teacherDataPath = "C:\\Users\\odyssey8942\\Documents\\get_car_data\\TeacherData\\ (1).csv";
    string orderlinessTeacherDataPath = "C:\\Users\\odyssey8942\\Documents\\get_car_data\\OrderlinessTeacherData\\ (";
    

    List<int> numbers = new List<int>();
    // Use this for initialization
    void Start () {
        List<int> numbers = new List<int>();

        for (int i = start; i <= end; i++)
        {
            numbers.Add(i);
        }
        j = 0;
        while (count-- > 0)
        {

            int index = UnityEngine.Random.Range(0, numbers.Count);

            dataNum[j++] = numbers[index];
            //Debug.Log(dataNum[j++]);

            numbers.RemoveAt(index);
        }
        //Debug.Log("あ");
    }
	
	// Update is called once per frame
	void Update () {
        //Debug.Log("い");
        j = 0;
        for (i = 0; i < 540; i++)
        {
            ReadWriteFile(dataNum[i]);
            Debug.Log(i);
        }
        SceneManager.LoadScene("taskcompleted");
    }
    void ReadWriteFile(int dataNum)
    {
        var dataString = new List<string>();
        String filePath = "C:\\Users\\odyssey8942\\Documents\\get_car_data\\TeacherData\\ (" + dataNum + ").csv";
        //Debug.Log(filePath);
        //FileInfo fi = new FileInfo(filePath);
        try
        {
            // 一行毎読み込み
            using (var sr = new System.IO.StreamReader(@filePath))
            {
                // ストリームの末尾まで繰り返す
                while (!sr.EndOfStream)
                {
                    // ファイルから一行読み込む
                    var line = sr.ReadLine();
                    dataString.Add(line);
                }
                foreach (var item in dataString)
                {
                    Debug.Log(item);
                }
                //Debug.Log(dataString.Count);
            }
        }
        catch (Exception e)
        {

        }
        for (j = dataString.Count - 1; j >= 35; j-=20)
        {
            for(int l = 30; l > 10 ; l--)
            {
                File.AppendAllText(orderlinessTeacherDataPath + k + ").csv", dataString[j-l] + "\n");
            }
            File.AppendAllText(orderlinessTeacherDataPath + k + ").csv", dataString[j]);
            k++;
        }    
    }
}
